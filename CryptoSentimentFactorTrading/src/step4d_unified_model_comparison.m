%% ============================================================
% STEP 4D - UNIFIED MODEL COMPARISON
% Crypto Sentiment Factor Trading
%
% REGRESSION MODELS
%   1. AR(5)
%   2. LSBoost Regression
%   3. Bagged Regression
%
% CLASSIFICATION MODELS
%   4. AdaBoostM2
%   5. RUSBoost
%   6. Random Forest
%   7. SVM-ECOC
%
% COMMON DATA SPLIT
%   Training   : 70%
%   Validation : 15%
%   Test       : 15%
%
% IMPORTANT
%   - Test data is untouched during model selection.
%   - Regression models predict next-day return.
%   - Classification models predict BUY / HOLD / SELL.
%   - Trading performance is calculated for all models.
%% ============================================================

clear;
clc;
close all;

fprintf('\n=====================================================\n');
fprintf('       UNIFIED MODEL COMPARISON\n');
fprintf('=====================================================\n');

%% ============================================================
% PATHS
%% ============================================================

dataFile = '../data/processed/factor_dataset.csv';

resultsTablePath = ...
    '../results/tables/unified_model_comparison.csv';

resultsModelPath = ...
    '../results/models/unified_models.mat';

performanceFigurePath = ...
    '../results/figures/unified_model_performance.png';

tradingFigurePath = ...
    '../results/figures/unified_trading_performance.png';

%% ============================================================
% CREATE DIRECTORIES
%% ============================================================

if ~exist('../results','dir')
    mkdir('../results');
end

if ~exist('../results/tables','dir')
    mkdir('../results/tables');
end

if ~exist('../results/models','dir')
    mkdir('../results/models');
end

if ~exist('../results/figures','dir')
    mkdir('../results/figures');
end

%% ============================================================
% LOAD DATA
%% ============================================================

fprintf('\nLoading factor dataset...\n');

if ~isfile(dataFile)

    error( ...
        'Dataset not found: %s',...
        dataFile);

end

T = readtable(dataFile);

fprintf('Original rows    : %d\n',height(T));

fprintf('Original columns : %d\n',width(T));

%% ============================================================
% REQUIRED VARIABLES
%% ============================================================

requiredVars = {
    'TargetReturn1D'
    'Close'
};

for i = 1:numel(requiredVars)

    if ~ismember( ...
            requiredVars{i},...
            T.Properties.VariableNames)

        error( ...
            'Required variable not found: %s',...
            requiredVars{i});

    end

end

%% ============================================================
% TARGET CLEANING
%% ============================================================

targetReturn = T.TargetReturn1D;

validTarget = isfinite(targetReturn);

T = T(validTarget,:);

fprintf( ...
    'Rows after target cleaning : %d\n',...
    height(T));

%% ============================================================
% FEATURE SELECTION
%% ============================================================

fprintf('\nPreparing features...\n');

excludeVars = {
    'Date'
    'TargetReturn1D'
    'TargetDirection'
};

allVars = T.Properties.VariableNames;

featureNames = setdiff( ...
    allVars,...
    excludeVars,...
    'stable');

%% Keep numeric variables only

numericMask = false(size(featureNames));

for i = 1:numel(featureNames)

    numericMask(i) = ...
        isnumeric(T.(featureNames{i}));

end

featureNames = featureNames(numericMask);

Xall = T{:,featureNames};

%% ============================================================
% REMOVE INVALID FEATURE ROWS
%% ============================================================

validFeatures = all(isfinite(Xall),2);

fprintf( ...
    'Rows with valid features : %d\n',...
    sum(validFeatures));

fprintf( ...
    'Rows removed due to NaN/Inf : %d\n',...
    sum(~validFeatures));

T = T(validFeatures,:);

Xall = Xall(validFeatures,:);

targetReturn = T.TargetReturn1D;

closePrice = T.Close;

fprintf('\nFinal usable dataset:\n');

fprintf('Rows     : %d\n',height(T));

fprintf('Features : %d\n',size(Xall,2));

%% ============================================================
% CREATE BUY / HOLD / SELL TARGET
%% ============================================================

classificationThreshold = 0.009;

classTarget = strings(height(T),1);

classTarget( ...
    targetReturn > classificationThreshold) = "BUY";

classTarget( ...
    targetReturn < -classificationThreshold) = "SELL";

classTarget( ...
    targetReturn >= -classificationThreshold & ...
    targetReturn <= classificationThreshold) = "HOLD";

Yclass = categorical( ...
    classTarget,...
    ["SELL","HOLD","BUY"]);

%% ============================================================
% CLASS DISTRIBUTION
%% ============================================================

fprintf('\nClass distribution:\n');

classNames = ["SELL","HOLD","BUY"];

for i = 1:numel(classNames)

    nClass = sum( ...
        Yclass == classNames(i));

    fprintf( ...
        '%-6s : %4d (%.2f%%)\n',...
        classNames(i),...
        nClass,...
        100*nClass/height(T));

end

%% ============================================================
% COMMON TIME-ORDERED SPLIT
%% ============================================================

n = size(Xall,1);

nTrain = floor(0.70*n);

nVal = floor(0.15*n);

nTest = n - nTrain - nVal;

idxTrain = 1:nTrain;

idxVal = ...
    nTrain+1:nTrain+nVal;

idxTest = ...
    nTrain+nVal+1:n;

%% ============================================================
% DATASETS
%% ============================================================

Xtrain = Xall(idxTrain,:);

Xval = Xall(idxVal,:);

Xtest = Xall(idxTest,:);

Ytrain = Yclass(idxTrain);

Yval = Yclass(idxVal);

Ytest = Yclass(idxTest);

Rtrain = targetReturn(idxTrain);

Rval = targetReturn(idxVal);

Rtest = targetReturn(idxTest);

CloseTest = closePrice(idxTest);

fprintf('\n=====================================================\n');
fprintf(' COMMON DATA SPLIT\n');
fprintf('=====================================================\n');

fprintf('Training   : %d observations\n',nTrain);

fprintf('Validation : %d observations\n',nVal);

fprintf('Testing    : %d observations\n',nTest);

%% ============================================================
% DATE RANGES
%% ============================================================

if ismember('Date',T.Properties.VariableNames)

    fprintf('\nDate ranges:\n');

    try

        fprintf( ...
            'Training   : %s -> %s\n',...
            string(T.Date(idxTrain(1))),...
            string(T.Date(idxTrain(end))));

        fprintf( ...
            'Validation : %s -> %s\n',...
            string(T.Date(idxVal(1))),...
            string(T.Date(idxVal(end))));

        fprintf( ...
            'Testing    : %s -> %s\n',...
            string(T.Date(idxTest(1))),...
            string(T.Date(idxTest(end))));

    catch

        fprintf( ...
            'Date ranges could not be displayed.\n');

    end

end

%% ============================================================
% STANDARDIZATION FOR SVM
%% ============================================================

mu = mean(Xtrain,1);

sigma = std(Xtrain,0,1);

sigma(sigma == 0) = 1;

XtrainSVM = ...
    (Xtrain - mu) ./ sigma;

XvalSVM = ...
    (Xval - mu) ./ sigma;

XtestSVM = ...
    (Xtest - mu) ./ sigma;

%% ============================================================
% RESULT STORAGE
%% ============================================================

numModels = 7;

ModelNames = strings(numModels,1);

ModelType = strings(numModels,1);

ValRMSE = nan(numModels,1);

ValMAE = nan(numModels,1);

ValR2 = nan(numModels,1);

ValDirectionalAccuracy = nan(numModels,1);

ValAccuracy = nan(numModels,1);

ValMacroF1 = nan(numModels,1);

ValBalancedAccuracy = nan(numModels,1);

TestRMSE = nan(numModels,1);

TestMAE = nan(numModels,1);

TestR2 = nan(numModels,1);

TestDirectionalAccuracy = nan(numModels,1);

TestAccuracy = nan(numModels,1);

TestMacroF1 = nan(numModels,1);

TestBalancedAccuracy = nan(numModels,1);

StrategyReturn = nan(numModels,1);

SharpeRatio = nan(numModels,1);

MaxDrawdown = nan(numModels,1);

WinRate = nan(numModels,1);

NumTrades = nan(numModels,1);

FinalCapital = nan(numModels,1);

Models = struct();

%% ============================================================
% TRADING PARAMETERS
%% ============================================================

initialCapital = 10000;

transactionCost = 0.001;

signalThreshold = 0.009;

%% ============================================================
% MODEL 1 - AR(5)
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 1: AR(5) REGRESSION\n');
fprintf('=====================================================\n');

fprintf('Training AR(5)...\n');

arOrder = 5;

% AR model uses lagged target returns.
% Training observations start after lag requirement.

if numel(Rtrain) <= arOrder

    error('Not enough observations for AR(5).');

end

ARXtrain = zeros( ...
    numel(Rtrain)-arOrder,...
    arOrder);

ARYtrain = zeros( ...
    numel(Rtrain)-arOrder,...
    1);

for i = arOrder+1:numel(Rtrain)

    ARXtrain( ...
        i-arOrder,:) = ...
        Rtrain(i-1:-1:i-arOrder);

    ARYtrain( ...
        i-arOrder) = Rtrain(i);

end

ARModel = fitrlinear( ...
    ARXtrain,...
    ARYtrain,...
    'Learner','leastsquares',...
    'Regularization','ridge');

%% Validation AR

RvalHistory = [Rtrain; Rval];

ARXval = zeros( ...
    numel(Rval),...
    arOrder);

for i = 1:numel(Rval)

    globalIndex = ...
        numel(Rtrain) + i;

    ARXval(i,:) = ...
        RvalHistory( ...
        globalIndex-1:-1:globalIndex-arOrder);

end

PredVal_AR = ...
    predict(ARModel,ARXval);

%% Test AR

Rhistory = ...
    [Rtrain; Rval; Rtest];

ARXtest = zeros( ...
    numel(Rtest),...
    arOrder);

for i = 1:numel(Rtest)

    globalIndex = ...
        numel(Rtrain)+numel(Rval)+i;

    ARXtest(i,:) = ...
        Rhistory( ...
        globalIndex-1:-1:globalIndex-arOrder);

end

PredTest_AR = ...
    predict(ARModel,ARXtest);

%% Regression metrics

[rmse,mae,r2] = ...
    regressionMetrics(Rval,PredVal_AR);

ValRMSE(1) = rmse;

ValMAE(1) = mae;

ValR2(1) = r2;

ValDirectionalAccuracy(1) = ...
    directionalAccuracy(Rval,PredVal_AR);

[rmse,mae,r2] = ...
    regressionMetrics(Rtest,PredTest_AR);

TestRMSE(1) = rmse;

TestMAE(1) = mae;

TestR2(1) = r2;

TestDirectionalAccuracy(1) = ...
    directionalAccuracy(Rtest,PredTest_AR);

%% Convert regression prediction to trading signal

SignalTest_AR = ...
    returnToSignal( ...
    PredTest_AR,...
    signalThreshold);

[trRet,sharpe,dd,win,trades,finalCap] = ...
    backtestStrategy( ...
    Rtest,...
    SignalTest_AR,...
    initialCapital,...
    transactionCost);

StrategyReturn(1) = trRet;

SharpeRatio(1) = sharpe;

MaxDrawdown(1) = dd;

WinRate(1) = win;

NumTrades(1) = trades;

FinalCapital(1) = finalCap;

ModelNames(1) = "AR(5)";

ModelType(1) = "Regression";

Models.AR5 = ARModel;

fprintf('Validation RMSE : %.6f\n',ValRMSE(1));

fprintf('Validation MAE  : %.6f\n',ValMAE(1));

fprintf('Validation R2   : %.6f\n',ValR2(1));

fprintf('Test RMSE       : %.6f\n',TestRMSE(1));

fprintf('Test MAE        : %.6f\n',TestMAE(1));

fprintf('Test R2         : %.6f\n',TestR2(1));

fprintf('Test Directional Accuracy : %.2f %%\n',...
    100*TestDirectionalAccuracy(1));

fprintf('Strategy Return : %.2f %%\n',...
    100*StrategyReturn(1));

fprintf('Sharpe Ratio    : %.4f\n',...
    SharpeRatio(1));

fprintf('Max Drawdown    : %.2f %%\n',...
    100*MaxDrawdown(1));

fprintf('Final Capital   : $%.2f\n',...
    FinalCapital(1));

%% ============================================================
% MODEL 2 - LSBOOST REGRESSION
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 2: LSBOOST REGRESSION\n');
fprintf('=====================================================\n');

fprintf('Training LSBoost Regression...\n');

LSModel = fitrensemble( ...
    Xtrain,...
    Rtrain,...
    'Method','LSBoost',...
    'NumLearningCycles',200,...
    'Learners',templateTree( ...
        'MaxNumSplits',20));

PredVal_LS = ...
    predict(LSModel,Xval);

PredTest_LS = ...
    predict(LSModel,Xtest);

[rmse,mae,r2] = ...
    regressionMetrics(Rval,PredVal_LS);

ValRMSE(2) = rmse;

ValMAE(2) = mae;

ValR2(2) = r2;

ValDirectionalAccuracy(2) = ...
    directionalAccuracy(Rval,PredVal_LS);

[rmse,mae,r2] = ...
    regressionMetrics(Rtest,PredTest_LS);

TestRMSE(2) = rmse;

TestMAE(2) = mae;

TestR2(2) = r2;

TestDirectionalAccuracy(2) = ...
    directionalAccuracy(Rtest,PredTest_LS);

SignalTest_LS = ...
    returnToSignal( ...
    PredTest_LS,...
    signalThreshold);

[trRet,sharpe,dd,win,trades,finalCap] = ...
    backtestStrategy( ...
    Rtest,...
    SignalTest_LS,...
    initialCapital,...
    transactionCost);

StrategyReturn(2) = trRet;

SharpeRatio(2) = sharpe;

MaxDrawdown(2) = dd;

WinRate(2) = win;

NumTrades(2) = trades;

FinalCapital(2) = finalCap;

ModelNames(2) = "LSBoost";

ModelType(2) = "Regression";

Models.LSBoost = LSModel;

fprintf('Validation RMSE : %.6f\n',ValRMSE(2));

fprintf('Validation MAE  : %.6f\n',ValMAE(2));

fprintf('Validation R2   : %.6f\n',ValR2(2));

fprintf('Test RMSE       : %.6f\n',TestRMSE(2));

fprintf('Test MAE        : %.6f\n',TestMAE(2));

fprintf('Test R2         : %.6f\n',TestR2(2));

fprintf('Test Directional Accuracy : %.2f %%\n',...
    100*TestDirectionalAccuracy(2));

fprintf('Strategy Return : %.2f %%\n',...
    100*StrategyReturn(2));

fprintf('Sharpe Ratio    : %.4f\n',...
    SharpeRatio(2));

fprintf('Max Drawdown    : %.2f %%\n',...
    100*MaxDrawdown(2));

fprintf('Final Capital   : $%.2f\n',...
    FinalCapital(2));

%% ============================================================
% MODEL 3 - BAGGED REGRESSION
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 3: BAGGED REGRESSION\n');
fprintf('=====================================================\n');

fprintf('Training Bagged Regression...\n');

BagRegModel = fitrensemble( ...
    Xtrain,...
    Rtrain,...
    'Method','Bag',...
    'NumLearningCycles',300,...
    'Learners',templateTree( ...
        'MaxNumSplits',20));

PredVal_BagReg = ...
    predict(BagRegModel,Xval);

PredTest_BagReg = ...
    predict(BagRegModel,Xtest);

[rmse,mae,r2] = ...
    regressionMetrics(Rval,PredVal_BagReg);

ValRMSE(3) = rmse;

ValMAE(3) = mae;

ValR2(3) = r2;

ValDirectionalAccuracy(3) = ...
    directionalAccuracy(Rval,PredVal_BagReg);

[rmse,mae,r2] = ...
    regressionMetrics(Rtest,PredTest_BagReg);

TestRMSE(3) = rmse;

TestMAE(3) = mae;

TestR2(3) = r2;

TestDirectionalAccuracy(3) = ...
    directionalAccuracy(Rtest,PredTest_BagReg);

SignalTest_BagReg = ...
    returnToSignal( ...
    PredTest_BagReg,...
    signalThreshold);

[trRet,sharpe,dd,win,trades,finalCap] = ...
    backtestStrategy( ...
    Rtest,...
    SignalTest_BagReg,...
    initialCapital,...
    transactionCost);

StrategyReturn(3) = trRet;

SharpeRatio(3) = sharpe;

MaxDrawdown(3) = dd;

WinRate(3) = win;

NumTrades(3) = trades;

FinalCapital(3) = finalCap;

ModelNames(3) = "Bagged Regression";

ModelType(3) = "Regression";

Models.BaggedRegression = BagRegModel;

fprintf('Validation RMSE : %.6f\n',ValRMSE(3));

fprintf('Validation MAE  : %.6f\n',ValMAE(3));

fprintf('Validation R2   : %.6f\n',ValR2(3));

fprintf('Test RMSE       : %.6f\n',TestRMSE(3));

fprintf('Test MAE        : %.6f\n',TestMAE(3));

fprintf('Test R2         : %.6f\n',TestR2(3));

fprintf('Test Directional Accuracy : %.2f %%\n',...
    100*TestDirectionalAccuracy(3));

fprintf('Strategy Return : %.2f %%\n',...
    100*StrategyReturn(3));

fprintf('Sharpe Ratio    : %.4f\n',...
    SharpeRatio(3));

fprintf('Max Drawdown    : %.2f %%\n',...
    100*MaxDrawdown(3));

fprintf('Final Capital   : $%.2f\n',...
    FinalCapital(3));

%% ============================================================
% MODEL 4 - ADABOOST M2
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 4: ADABOOST M2\n');
fprintf('=====================================================\n');

fprintf('Training AdaBoostM2...\n');

AdaModel = fitcensemble( ...
    Xtrain,...
    Ytrain,...
    'Method','AdaBoostM2',...
    'NumLearningCycles',200);

PredVal_Ada = ...
    predict(AdaModel,Xval);

PredTest_Ada = ...
    predict(AdaModel,Xtest);

[valAcc,valF1,valBal] = ...
    evaluateClassification( ...
    Yval,...
    PredVal_Ada);

ValAccuracy(4) = valAcc;

ValMacroF1(4) = valF1;

ValBalancedAccuracy(4) = valBal;

[testAcc,testF1,testBal] = ...
    evaluateClassification( ...
    Ytest,...
    PredTest_Ada);

TestAccuracy(4) = testAcc;

TestMacroF1(4) = testF1;

TestBalancedAccuracy(4) = testBal;

TestDirectionalAccuracy(4) = ...
    classificationDirectionalAccuracy( ...
    Rtest,...
    PredTest_Ada);

SignalTest_Ada = ...
    classificationToSignal(PredTest_Ada);

[trRet,sharpe,dd,win,trades,finalCap] = ...
    backtestStrategy( ...
    Rtest,...
    SignalTest_Ada,...
    initialCapital,...
    transactionCost);

StrategyReturn(4) = trRet;

SharpeRatio(4) = sharpe;

MaxDrawdown(4) = dd;

WinRate(4) = win;

NumTrades(4) = trades;

FinalCapital(4) = finalCap;

ModelNames(4) = "AdaBoostM2";

ModelType(4) = "Classification";

Models.AdaBoostM2 = AdaModel;

fprintf('Validation Accuracy : %.2f %%\n',100*valAcc);

fprintf('Validation Macro F1 : %.2f %%\n',100*valF1);

fprintf('Validation Balanced Accuracy : %.2f %%\n',100*valBal);

fprintf('Test Accuracy : %.2f %%\n',100*testAcc);

fprintf('Test Macro F1 : %.2f %%\n',100*testF1);

fprintf('Test Balanced Accuracy : %.2f %%\n',100*testBal);

fprintf('Strategy Return : %.2f %%\n',100*StrategyReturn(4));

fprintf('Sharpe Ratio : %.4f\n',SharpeRatio(4));

fprintf('Max Drawdown : %.2f %%\n',100*MaxDrawdown(4));

fprintf('Final Capital : $%.2f\n',FinalCapital(4));

%% ============================================================
% MODEL 5 - RUSBOOST
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 5: RUSBOOST\n');
fprintf('=====================================================\n');

fprintf('Training RUSBoost...\n');

RusModel = fitcensemble( ...
    Xtrain,...
    Ytrain,...
    'Method','RUSBoost',...
    'NumLearningCycles',200);

PredVal_Rus = ...
    predict(RusModel,Xval);

PredTest_Rus = ...
    predict(RusModel,Xtest);

[valAcc,valF1,valBal] = ...
    evaluateClassification( ...
    Yval,...
    PredVal_Rus);

ValAccuracy(5) = valAcc;

ValMacroF1(5) = valF1;

ValBalancedAccuracy(5) = valBal;

[testAcc,testF1,testBal] = ...
    evaluateClassification( ...
    Ytest,...
    PredTest_Rus);

TestAccuracy(5) = testAcc;

TestMacroF1(5) = testF1;

TestBalancedAccuracy(5) = testBal;

TestDirectionalAccuracy(5) = ...
    classificationDirectionalAccuracy( ...
    Rtest,...
    PredTest_Rus);

SignalTest_Rus = ...
    classificationToSignal(PredTest_Rus);

[trRet,sharpe,dd,win,trades,finalCap] = ...
    backtestStrategy( ...
    Rtest,...
    SignalTest_Rus,...
    initialCapital,...
    transactionCost);

StrategyReturn(5) = trRet;

SharpeRatio(5) = sharpe;

MaxDrawdown(5) = dd;

WinRate(5) = win;

NumTrades(5) = trades;

FinalCapital(5) = finalCap;

ModelNames(5) = "RUSBoost";

ModelType(5) = "Classification";

Models.RUSBoost = RusModel;

fprintf('Validation Accuracy : %.2f %%\n',100*valAcc);

fprintf('Validation Macro F1 : %.2f %%\n',100*valF1);

fprintf('Validation Balanced Accuracy : %.2f %%\n',100*valBal);

fprintf('Test Accuracy : %.2f %%\n',100*testAcc);

fprintf('Test Macro F1 : %.2f %%\n',100*testF1);

fprintf('Test Balanced Accuracy : %.2f %%\n',100*testBal);

fprintf('Strategy Return : %.2f %%\n',100*StrategyReturn(5));

fprintf('Sharpe Ratio : %.4f\n',SharpeRatio(5));

fprintf('Max Drawdown : %.2f %%\n',100*MaxDrawdown(5));

fprintf('Final Capital : $%.2f\n',FinalCapital(5));

%% ============================================================
% MODEL 6 - RANDOM FOREST
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 6: RANDOM FOREST\n');
fprintf('=====================================================\n');

fprintf('Training Random Forest...\n');

RFModel = TreeBagger( ...
    300,...
    Xtrain,...
    Ytrain,...
    'Method','classification',...
    'OOBPrediction','On');

PredVal_RF = ...
    predict(RFModel,Xval);

PredVal_RF = categorical(PredVal_RF);

PredTest_RF = ...
    predict(RFModel,Xtest);

PredTest_RF = categorical(PredTest_RF);

[valAcc,valF1,valBal] = ...
    evaluateClassification( ...
    Yval,...
    PredVal_RF);

ValAccuracy(6) = valAcc;

ValMacroF1(6) = valF1;

ValBalancedAccuracy(6) = valBal;

[testAcc,testF1,testBal] = ...
    evaluateClassification( ...
    Ytest,...
    PredTest_RF);

TestAccuracy(6) = testAcc;

TestMacroF1(6) = testF1;

TestBalancedAccuracy(6) = testBal;

TestDirectionalAccuracy(6) = ...
    classificationDirectionalAccuracy( ...
    Rtest,...
    PredTest_RF);

SignalTest_RF = ...
    classificationToSignal(PredTest_RF);

[trRet,sharpe,dd,win,trades,finalCap] = ...
    backtestStrategy( ...
    Rtest,...
    SignalTest_RF,...
    initialCapital,...
    transactionCost);

StrategyReturn(6) = trRet;

SharpeRatio(6) = sharpe;

MaxDrawdown(6) = dd;

WinRate(6) = win;

NumTrades(6) = trades;

FinalCapital(6) = finalCap;

ModelNames(6) = "Random Forest";

ModelType(6) = "Classification";

Models.RandomForest = RFModel;

fprintf('Validation Accuracy : %.2f %%\n',100*valAcc);

fprintf('Validation Macro F1 : %.2f %%\n',100*valF1);

fprintf('Validation Balanced Accuracy : %.2f %%\n',100*valBal);

fprintf('Test Accuracy : %.2f %%\n',100*testAcc);

fprintf('Test Macro F1 : %.2f %%\n',100*testF1);

fprintf('Test Balanced Accuracy : %.2f %%\n',100*testBal);

fprintf('Strategy Return : %.2f %%\n',100*StrategyReturn(6));

fprintf('Sharpe Ratio : %.4f\n',SharpeRatio(6));

fprintf('Max Drawdown : %.2f %%\n',100*MaxDrawdown(6));

fprintf('Final Capital : $%.2f\n',FinalCapital(6));

%% ============================================================
% MODEL 7 - SVM ECOC
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 7: SVM-ECOC\n');
fprintf('=====================================================\n');

fprintf('Training SVM-ECOC...\n');

SVMModel = fitcecoc( ...
    XtrainSVM,...
    Ytrain,...
    'Learners','linear');

PredVal_SVM = ...
    predict(SVMModel,XvalSVM);

PredTest_SVM = ...
    predict(SVMModel,XtestSVM);

[valAcc,valF1,valBal] = ...
    evaluateClassification( ...
    Yval,...
    PredVal_SVM);

ValAccuracy(7) = valAcc;

ValMacroF1(7) = valF1;

ValBalancedAccuracy(7) = valBal;

[testAcc,testF1,testBal] = ...
    evaluateClassification( ...
    Ytest,...
    PredTest_SVM);

TestAccuracy(7) = testAcc;

TestMacroF1(7) = testF1;

TestBalancedAccuracy(7) = testBal;

TestDirectionalAccuracy(7) = ...
    classificationDirectionalAccuracy( ...
    Rtest,...
    PredTest_SVM);

SignalTest_SVM = ...
    classificationToSignal(PredTest_SVM);

[trRet,sharpe,dd,win,trades,finalCap] = ...
    backtestStrategy( ...
    Rtest,...
    SignalTest_SVM,...
    initialCapital,...
    transactionCost);

StrategyReturn(7) = trRet;

SharpeRatio(7) = sharpe;

MaxDrawdown(7) = dd;

WinRate(7) = win;

NumTrades(7) = trades;

FinalCapital(7) = finalCap;

ModelNames(7) = "SVM-ECOC";

ModelType(7) = "Classification";

Models.SVM_ECOC = SVMModel;

fprintf('Validation Accuracy : %.2f %%\n',100*valAcc);

fprintf('Validation Macro F1 : %.2f %%\n',100*valF1);

fprintf('Validation Balanced Accuracy : %.2f %%\n',100*valBal);

fprintf('Test Accuracy : %.2f %%\n',100*testAcc);

fprintf('Test Macro F1 : %.2f %%\n',100*testF1);

fprintf('Test Balanced Accuracy : %.2f %%\n',100*testBal);

fprintf('Strategy Return : %.2f %%\n',100*StrategyReturn(7));

fprintf('Sharpe Ratio : %.4f\n',SharpeRatio(7));

fprintf('Max Drawdown : %.2f %%\n',100*MaxDrawdown(7));

fprintf('Final Capital : $%.2f\n',FinalCapital(7));

%% ============================================================
% BUY AND HOLD BENCHMARK
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' BUY & HOLD BENCHMARK\n');
fprintf('=====================================================\n');

buyHoldReturns = Rtest;

buyHoldEquity = ...
    initialCapital * ...
    cumprod(1 + buyHoldReturns);

buyHoldFinal = ...
    buyHoldEquity(end);

buyHoldReturn = ...
    buyHoldFinal/initialCapital - 1;

buyHoldSharpe = ...
    calculateSharpe(buyHoldReturns);

buyHoldDD = ...
    calculateMaxDrawdown(buyHoldEquity);

fprintf('Final Capital : $%.2f\n',buyHoldFinal);

fprintf('Return        : %.2f %%\n',100*buyHoldReturn);

fprintf('Sharpe        : %.4f\n',buyHoldSharpe);

fprintf('Max Drawdown  : %.2f %%\n',100*buyHoldDD);

%% ============================================================
% UNIFIED RESULTS TABLE
%% ============================================================

Results = table( ...
    ModelNames,...
    ModelType,...
    ValRMSE,...
    ValMAE,...
    ValR2,...
    100*ValDirectionalAccuracy,...
    100*ValAccuracy,...
    100*ValMacroF1,...
    100*ValBalancedAccuracy,...
    TestRMSE,...
    TestMAE,...
    TestR2,...
    100*TestDirectionalAccuracy,...
    100*TestAccuracy,...
    100*TestMacroF1,...
    100*TestBalancedAccuracy,...
    100*StrategyReturn,...
    SharpeRatio,...
    100*MaxDrawdown,...
    100*WinRate,...
    NumTrades,...
    FinalCapital,...
    'VariableNames',{ ...
    'Model',...
    'Type',...
    'ValidationRMSE',...
    'ValidationMAE',...
    'ValidationR2',...
    'ValidationDirectionalAccuracyPct',...
    'ValidationAccuracyPct',...
    'ValidationMacroF1Pct',...
    'ValidationBalancedAccuracyPct',...
    'TestRMSE',...
    'TestMAE',...
    'TestR2',...
    'TestDirectionalAccuracyPct',...
    'TestAccuracyPct',...
    'TestMacroF1Pct',...
    'TestBalancedAccuracyPct',...
    'StrategyReturnPct',...
    'SharpeRatio',...
    'MaxDrawdownPct',...
    'WinRatePct',...
    'NumberOfTrades',...
    'FinalCapital'});

%% ============================================================
% DISPLAY RESULTS
%% ============================================================

fprintf('\n\n=====================================================\n');
fprintf('              UNIFIED MODEL RESULTS\n');
fprintf('=====================================================\n\n');

disp(Results);

%% ============================================================
% MODEL SELECTION - VALIDATION DATA
%
% For regression models:
%   Lower RMSE is better.
%
% For classification models:
%   Higher Macro F1 is better.
%
% We therefore report separate validation selections.
%% ============================================================

regressionIdx = ...
    ModelType == "Regression";

classificationIdx = ...
    ModelType == "Classification";

%% Best regression based on validation RMSE

regIndices = find(regressionIdx);

[~,localBestReg] = ...
    min(ValRMSE(regIndices));

bestRegressionIdx = ...
    regIndices(localBestReg);

%% Best classification based on validation Macro F1

classIndices = find(classificationIdx);

[~,localBestClass] = ...
    max(ValMacroF1(classIndices));

bestClassificationIdx = ...
    classIndices(localBestClass);

fprintf('\n=====================================================\n');
fprintf(' VALIDATION-BASED MODEL SELECTION\n');
fprintf('=====================================================\n');

fprintf('\nBest regression model by validation RMSE:\n');

fprintf('  %s\n',...
    Results.Model(bestRegressionIdx));

fprintf('  Validation RMSE : %.6f\n',...
    Results.ValidationRMSE(bestRegressionIdx));

fprintf('  Test RMSE       : %.6f\n',...
    Results.TestRMSE(bestRegressionIdx));

fprintf('\nBest classification model by validation Macro F1:\n');

fprintf('  %s\n',...
    Results.Model(bestClassificationIdx));

fprintf('  Validation Macro F1 : %.2f %%\n',...
    Results.ValidationMacroF1Pct(bestClassificationIdx));

fprintf('  Test Macro F1       : %.2f %%\n',...
    Results.TestMacroF1Pct(bestClassificationIdx));

%% ============================================================
% TRADING PERFORMANCE TABLE
%% ============================================================

TradingResults = table( ...
    Results.Model,...
    Results.Type,...
    Results.StrategyReturnPct,...
    Results.SharpeRatio,...
    Results.MaxDrawdownPct,...
    Results.WinRatePct,...
    Results.NumberOfTrades,...
    Results.FinalCapital,...
    'VariableNames',{ ...
    'Model',...
    'Type',...
    'StrategyReturnPct',...
    'SharpeRatio',...
    'MaxDrawdownPct',...
    'WinRatePct',...
    'NumberOfTrades',...
    'FinalCapital'});

fprintf('\n=====================================================\n');
fprintf('              TRADING PERFORMANCE\n');
fprintf('=====================================================\n\n');

disp(TradingResults);

%% ============================================================
% ADD BUY & HOLD TO DISPLAY
%% ============================================================

fprintf('\nBUY & HOLD:\n');

fprintf('  Return       : %.2f %%\n',100*buyHoldReturn);

fprintf('  Sharpe       : %.4f\n',buyHoldSharpe);

fprintf('  Max Drawdown : %.2f %%\n',100*buyHoldDD);

fprintf('  Final Capital: $%.2f\n',buyHoldFinal);

%% ============================================================
% SAVE RESULTS
%% ============================================================

writetable( ...
    Results,...
    resultsTablePath);

save( ...
    resultsModelPath,...
    'Models',...
    'Results',...
    'TradingResults',...
    'featureNames',...
    'mu',...
    'sigma',...
    'classificationThreshold',...
    'signalThreshold',...
    'initialCapital',...
    'transactionCost',...
    'bestRegressionIdx',...
    'bestClassificationIdx');

fprintf('\nResults saved:\n');

fprintf('  %s\n',resultsTablePath);

fprintf('  %s\n',resultsModelPath);

%% ============================================================
% PREDICTION PERFORMANCE GRAPH
%% ============================================================

figure( ...
    'Name','Unified Prediction Performance',...
    'Color','w');

subplot(1,2,1);

bar( ...
    1:numModels,...
    TestDirectionalAccuracy*100);

grid on;

xlabel('Model');

ylabel('Directional Accuracy (%)');

title('Test Directional Accuracy');

set(gca,...
    'XTick',1:numModels,...
    'XTickLabel',ModelNames);

xtickangle(30);

subplot(1,2,2);

bar( ...
    1:numModels,...
    TestMacroF1*100);

grid on;

xlabel('Model');

ylabel('Macro F1 (%)');

title('Test Macro F1');

set(gca,...
    'XTick',1:numModels,...
    'XTickLabel',ModelNames);

xtickangle(30);

saveas( ...
    gcf,...
    performanceFigurePath);

%% ============================================================
% TRADING PERFORMANCE GRAPH
%% ============================================================

figure( ...
    'Name','Unified Trading Performance',...
    'Color','w');

subplot(1,2,1);

bar( ...
    1:numModels,...
    StrategyReturn*100);

hold on;

yline( ...
    buyHoldReturn*100,...
    '--',...
    'Buy & Hold');

grid on;

xlabel('Model');

ylabel('Return (%)');

title('Strategy Return');

set(gca,...
    'XTick',1:numModels,...
    'XTickLabel',ModelNames);

xtickangle(30);

hold off;

subplot(1,2,2);

bar( ...
    1:numModels,...
    SharpeRatio);

hold on;

yline( ...
    buyHoldSharpe,...
    '--',...
    'Buy & Hold');

grid on;

xlabel('Model');

ylabel('Sharpe Ratio');

title('Sharpe Ratio');

set(gca,...
    'XTick',1:numModels,...
    'XTickLabel',ModelNames);

xtickangle(30);

hold off;

saveas( ...
    gcf,...
    tradingFigurePath);

%% ============================================================
% FINAL MESSAGE
%% ============================================================

fprintf('\n=====================================================\n');
fprintf('       UNIFIED EXPERIMENT COMPLETED\n');
fprintf('=====================================================\n');

fprintf('\nModels evaluated: %d\n',numModels);

fprintf('Regression models: 3\n');

fprintf('Classification models: 4\n');

fprintf('\nRegression selection:\n');

fprintf('  %s\n',Results.Model(bestRegressionIdx));

fprintf('\nClassification selection:\n');

fprintf('  %s\n',Results.Model(bestClassificationIdx));

fprintf('\nIMPORTANT:\n');

fprintf(['Do not interpret the regression RMSE and classification ',...
    'Macro F1 as directly comparable scores.\n']);

fprintf(['Use the common trading metrics for cross-model ',...
    'strategy comparison.\n']);

fprintf('\n=====================================================\n');


%% ============================================================
% LOCAL FUNCTION 1
% REGRESSION METRICS
%% ============================================================

function [rmse,mae,r2] = ...
    regressionMetrics(yTrue,yPred)

    yTrue = yTrue(:);

    yPred = yPred(:);

    valid = ...
        isfinite(yTrue) & ...
        isfinite(yPred);

    yTrue = yTrue(valid);

    yPred = yPred(valid);

    errors = yTrue - yPred;

    rmse = ...
        sqrt(mean(errors.^2));

    mae = ...
        mean(abs(errors));

    ssRes = ...
        sum(errors.^2);

    ssTot = ...
        sum((yTrue - mean(yTrue)).^2);

    if ssTot > 0

        r2 = ...
            1 - ssRes/ssTot;

    else

        r2 = NaN;

    end

end


%% ============================================================
% LOCAL FUNCTION 2
% DIRECTIONAL ACCURACY
%% ============================================================

function acc = ...
    directionalAccuracy(yTrue,yPred)

    yTrue = yTrue(:);

    yPred = yPred(:);

    valid = ...
        isfinite(yTrue) & ...
        isfinite(yPred);

    yTrue = yTrue(valid);

    yPred = yPred(valid);

    if isempty(yTrue)

        acc = NaN;

        return;

    end

    acc = ...
        mean(sign(yTrue) == sign(yPred));

end


%% ============================================================
% LOCAL FUNCTION 3
% CLASSIFICATION METRICS
%% ============================================================

function [accuracy,macroF1,balancedAccuracy] = ...
    evaluateClassification(Ytrue,Ypred)

    classNames = ["SELL","HOLD","BUY"];

    Ytrue = categorical( ...
        string(Ytrue),...
        classNames);

    Ypred = categorical( ...
        string(Ypred),...
        classNames);

    accuracy = ...
        mean(Ytrue == Ypred);

    classOrder = categorical( ...
        classNames,...
        classNames);

    C = confusionmat( ...
        Ytrue,...
        Ypred,...
        'Order',classOrder);

    nClasses = numel(classNames);

    recall = zeros(nClasses,1);

    F1 = zeros(nClasses,1);

    for k = 1:nClasses

        TP = C(k,k);

        FP = sum(C(:,k)) - TP;

        FN = sum(C(k,:)) - TP;

        if TP + FP > 0

            precision = ...
                TP/(TP+FP);

        else

            precision = 0;

        end

        if TP + FN > 0

            recall(k) = ...
                TP/(TP+FN);

        else

            recall(k) = 0;

        end

        if precision + recall(k) > 0

            F1(k) = ...
                2*precision*recall(k)/...
                (precision+recall(k));

        else

            F1(k) = 0;

        end

    end

    macroF1 = mean(F1);

    balancedAccuracy = mean(recall);

end


%% ============================================================
% LOCAL FUNCTION 4
% CLASSIFICATION DIRECTIONAL ACCURACY
%% ============================================================

function acc = ...
    classificationDirectionalAccuracy( ...
    trueReturns,predictedClasses)

    trueReturns = trueReturns(:);

    predictedClasses = categorical( ...
        string(predictedClasses),...
        ["SELL","HOLD","BUY"]);

    n = min( ...
        numel(trueReturns),...
        numel(predictedClasses));

    trueReturns = trueReturns(1:n);

    predictedClasses = ...
        predictedClasses(1:n);

    valid = isfinite(trueReturns);

    trueReturns = trueReturns(valid);

    predictedClasses = ...
        predictedClasses(valid);

    predictedDirection = zeros( ...
        numel(predictedClasses),1);

    predictedDirection( ...
        predictedClasses == "BUY") = 1;

    predictedDirection( ...
        predictedClasses == "SELL") = -1;

    trueDirection = sign(trueReturns);

    %% HOLD predictions are considered neutral.

    acc = ...
        mean(predictedDirection == trueDirection);

end


%% ============================================================
% LOCAL FUNCTION 5
% RETURN -> TRADING SIGNAL
%
% +1 = BUY
%  0 = HOLD
% -1 = SELL
%% ============================================================

function signal = ...
    returnToSignal(predictedReturn,threshold)

    signal = zeros(size(predictedReturn));

    signal(predictedReturn > threshold) = 1;

    signal(predictedReturn < -threshold) = -1;

end


%% ============================================================
% LOCAL FUNCTION 6
% CLASSIFICATION -> TRADING SIGNAL
%% ============================================================

function signal = ...
    classificationToSignal(predictedClass)

    predictedClass = categorical( ...
        string(predictedClass),...
        ["SELL","HOLD","BUY"]);

    signal = zeros( ...
        numel(predictedClass),1);

    signal(predictedClass == "BUY") = 1;

    signal(predictedClass == "SELL") = -1;

    signal(predictedClass == "HOLD") = 0;

end


%% ============================================================
% LOCAL FUNCTION 7
% BACKTEST
%% ============================================================

function [totalReturn,sharpe,maxDD,winRate,numTrades,finalCapital] = ...
    backtestStrategy( ...
    actualReturns,...
    signal,...
    initialCapital,...
    transactionCost)

    actualReturns = actualReturns(:);

    signal = signal(:);

    n = min( ...
        numel(actualReturns),...
        numel(signal));

    actualReturns = actualReturns(1:n);

    signal = signal(1:n);

    %% Position

    position = signal;

    %% Strategy return before costs

    strategyReturns = ...
        position .* actualReturns;

    %% Transaction costs

    previousPosition = ...
        [0;position(1:end-1)];

    turnover = ...
        abs(position - previousPosition);

    costs = ...
        turnover * transactionCost;

    netReturns = ...
        strategyReturns - costs;

    %% Equity curve

    equity = ...
        initialCapital * ...
        cumprod(1 + netReturns);

    finalCapital = equity(end);

    totalReturn = ...
        finalCapital/initialCapital - 1;

    %% Sharpe

    sharpe = ...
        calculateSharpe(netReturns);

    %% Maximum drawdown

    maxDD = ...
        calculateMaxDrawdown(equity);

    %% Trades

    tradeEvents = ...
        position ~= previousPosition;

    numTrades = ...
        sum(tradeEvents);

    %% Win rate

    activeTrades = ...
        position ~= 0;

    if sum(activeTrades) > 0

        wins = ...
            netReturns(activeTrades) > 0;

        winRate = ...
            mean(wins);

    else

        winRate = NaN;

    end

end


%% ============================================================
% LOCAL FUNCTION 8
% SHARPE RATIO
%% ============================================================

function sharpe = ...
    calculateSharpe(returns)

    returns = returns(:);

    returns = ...
        returns(isfinite(returns));

    if numel(returns) < 2

        sharpe = NaN;

        return;

    end

    s = std(returns);

    if s == 0

        sharpe = NaN;

        return;

    end

    sharpe = ...
        mean(returns)/s * sqrt(365);

end


%% ============================================================
% LOCAL FUNCTION 9
% MAXIMUM DRAWDOWN
%% ============================================================

function maxDD = ...
    calculateMaxDrawdown(equity)

    equity = equity(:);

    runningMaximum = ...
        cummax(equity);

    drawdown = ...
        equity./runningMaximum - 1;

    maxDD = ...
        min(drawdown);

end