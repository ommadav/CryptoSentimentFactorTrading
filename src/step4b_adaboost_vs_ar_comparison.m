function step4b_adaboost_vs_ar_comparison()
% ================================================================
% STEP 4B
% AR(5) vs CLASS-BALANCED ADABOOSTM2
% ================================================================
%
% Purpose:
%   Fair comparison between:
%       1. AR(5) autoregressive regression model
%       2. Class-Balanced AdaBoostM2 multiclass classifier
%
% Both models use:
%       - Same factor_dataset.csv
%       - Same chronological train/validation/test split
%       - Same 1-day target
%       - Same BUY/HOLD/SELL threshold
%       - Same transaction cost
%       - Same test period
%
% Outputs:
%       results/tables/ar_vs_adaboost_comparison.csv
%       results/tables/adaboost_classification_metrics.csv
%       results/tables/ar_adaboost_trading_metrics.csv
%
%       results/models/adaboost_m2_model.mat
%       results/models/ar5_model.mat
%       results/models/ar_vs_adaboost_comparison.mat
%
%       results/figures/ar_vs_adaboost_performance.png
%       results/figures/ar_vs_adaboost_returns.png
%       results/figures/ar_vs_adaboost_confusion.png
%
% ================================================================

clc;
fprintf('\n');
fprintf('============================================================\n');
fprintf('   AR(5) vs CLASS-BALANCED ADABOOSTM2 COMPARISON\n');
fprintf('============================================================\n\n');

%% ================================================================
% PROJECT PATH
% ================================================================

thisFile = mfilename('fullpath');

if isempty(thisFile)
    srcDir = pwd;
else
    srcDir = fileparts(thisFile);
end

projectDir = fileparts(srcDir);

processedDir = fullfile(projectDir,'data','processed');
resultsDir   = fullfile(projectDir,'results');

tablesDir = fullfile(resultsDir,'tables');
modelsDir = fullfile(resultsDir,'models');
figuresDir = fullfile(resultsDir,'figures');

if ~exist(tablesDir,'dir')
    mkdir(tablesDir);
end

if ~exist(modelsDir,'dir')
    mkdir(modelsDir);
end

if ~exist(figuresDir,'dir')
    mkdir(figuresDir);
end

factorFile = fullfile(processedDir,'factor_dataset.csv');

if ~isfile(factorFile)

    % Alternative location
    factorFile2 = fullfile(projectDir,'data','factor_dataset.csv');

    if isfile(factorFile2)
        factorFile = factorFile2;
    else
        error(['factor_dataset.csv not found.\n' ...
               'Expected location:\n%s\n\n' ...
               'Run Step 3 first.'],factorFile);
    end
end

fprintf('Dataset:\n%s\n\n',factorFile);

%% ================================================================
% SETTINGS
% ================================================================

AR_ORDER = 5;

% Same threshold used in your current AR(5) strategy
THRESHOLD = 0.009;

% Transaction cost = 0.10%
TRANSACTION_COST = 0.001;

% Initial capital
INITIAL_CAPITAL = 10000;

% Number of boosting learners
NUM_LEARNERS = 100;

% Maximum tree splits
MAX_SPLITS = 20;

rng(42,'twister');

%% ================================================================
% LOAD DATA
% ================================================================

fprintf('Loading factor dataset...\n');

data = readtable(factorFile);

fprintf('Rows loaded    : %d\n',height(data));
fprintf('Columns loaded : %d\n\n',width(data));

if height(data) < 100
    error('Dataset contains too few observations.');
end

%% ================================================================
% FIND REQUIRED COLUMNS
% ================================================================

targetCol = findColumn(data,...
    {'TargetReturn1D','TargetReturn','Target_1D','Target'});

if isempty(targetCol)
    error(['Target return column was not found.\n' ...
           'Expected TargetReturn1D or similar column.']);
end

dateCol = findColumn(data,{'Date','date','OpenTime','Open_time'});

closeCol = findColumn(data,{'Close','close'});

if isempty(closeCol)
    error('Close column was not found.');
end

fprintf('Target column : %s\n',targetCol);
fprintf('Close column  : %s\n',closeCol);

if isempty(dateCol)
    fprintf('Date column   : Not found\n\n');
else
    fprintf('Date column   : %s\n\n',dateCol);
end

%% ================================================================
% GET DATA
% ================================================================

target = getNumeric(data,targetCol);
closePrice = getNumeric(data,closeCol);

dates = getDates(data,dateCol);

%% ================================================================
% REMOVE INVALID TARGETS
% ================================================================

valid = isfinite(target) & isfinite(closePrice);

if ~isempty(dates)
    valid = valid & ~isnat(dates);
end

data = data(valid,:);

target = target(valid);
closePrice = closePrice(valid);

if ~isempty(dates)
    dates = dates(valid);
end

fprintf('Valid observations : %d\n\n',length(target));

%% ================================================================
% CREATE CLASS LABELS
% ================================================================

% Classification rule:
%
% target > +0.90%  -> BUY
% target < -0.90%  -> SELL
% otherwise        -> HOLD
%
% This gives the same decision threshold used by AR(5).

classLabels = strings(length(target),1);

classLabels(target > THRESHOLD) = "BUY";
classLabels(target < -THRESHOLD) = "SELL";
classLabels(target >= -THRESHOLD & target <= THRESHOLD) = "HOLD";

%% ================================================================
% CREATE NUMERIC CLASS LABELS
% ================================================================

% MATLAB categorical labels
YClass = categorical(classLabels,...
    ["SELL","HOLD","BUY"]);

%% ================================================================
% CHRONOLOGICAL SPLIT
% ================================================================

N = length(target);

trainN = min(830,N);

remaining = N-trainN;

valN = min(178,remaining);

testN = remaining-valN;

if testN <= 0
    error('Insufficient observations for test set.');
end

trainIdx = 1:trainN;

valIdx = trainN+1:trainN+valN;

testIdx = trainN+valN+1:N;

fprintf('============================================================\n');
fprintf('DATA SPLIT\n');
fprintf('============================================================\n');

fprintf('Total      : %d\n',N);
fprintf('Training   : %d\n',length(trainIdx));
fprintf('Validation : %d\n',length(valIdx));
fprintf('Test       : %d\n\n',length(testIdx));

if ~isempty(dates)

    fprintf('Training dates:\n');
    fprintf('  %s -> %s\n',...
        datestr(dates(trainIdx(1))),...
        datestr(dates(trainIdx(end))));

    fprintf('Validation dates:\n');
    fprintf('  %s -> %s\n',...
        datestr(dates(valIdx(1))),...
        datestr(dates(valIdx(end))));

    fprintf('Test dates:\n');
    fprintf('  %s -> %s\n\n',...
        datestr(dates(testIdx(1))),...
        datestr(dates(testIdx(end))));
end

%% ================================================================
% BUILD FEATURES FOR ADABOOST
% ================================================================

fprintf('Building AdaBoost features...\n');

X = buildAdaBoostFeatures(data,target,closePrice);

featureNames = X.Properties.VariableNames;

X = table2array(X);

X = double(X);

% Replace invalid values
for j = 1:size(X,2)

    col = X(:,j);

    finiteValues = col(isfinite(col));

    if isempty(finiteValues)
        replacement = 0;
    else
        replacement = median(finiteValues);
    end

    col(~isfinite(col)) = replacement;

    X(:,j) = col;
end

fprintf('AdaBoost features : %d\n',size(X,2));
fprintf('Feature names:\n');

for i=1:length(featureNames)
    fprintf('  %02d. %s\n',i,featureNames{i});
end

fprintf('\n');

%% ================================================================
% STANDARDIZE FEATURES
% ================================================================

% Scaling parameters calculated ONLY on training data.

mu = mean(X(trainIdx,:),1,'omitnan');

sigma = std(X(trainIdx,:),0,1,'omitnan');

sigma(sigma < eps | ~isfinite(sigma)) = 1;

Xscaled = (X-mu)./sigma;

%% ================================================================
% ADABOOST MODEL
% ================================================================

fprintf('============================================================\n');
fprintf('TRAINING CLASS-BALANCED ADABOOSTM2\n');
fprintf('============================================================\n');

XTrain = Xscaled(trainIdx,:);
YTrain = YClass(trainIdx);

XVal = Xscaled(valIdx,:);
YVal = YClass(valIdx);

XTest = Xscaled(testIdx,:);
YTest = YClass(testIdx);

%% ================================================================
% CLASS BALANCING
% ================================================================

trainClasses = categories(YTrain);

classWeights = ones(length(YTrain),1);

classCounts = zeros(length(trainClasses),1);

for i=1:length(trainClasses)

    idx = YTrain == trainClasses{i};

    classCounts(i)=sum(idx);

end

fprintf('Training class distribution:\n');

for i=1:length(trainClasses)

    fprintf('  %-5s : %d\n',...
        trainClasses{i},...
        classCounts(i));

end

fprintf('\n');

% Inverse-frequency weighting
for i=1:length(trainClasses)

    idx = YTrain == trainClasses{i};

    if classCounts(i)>0
        classWeights(idx)=length(YTrain)/(length(trainClasses)*classCounts(i));
    end

end

%% ================================================================
% CREATE TREE TEMPLATE
% ================================================================

treeTemplate = templateTree(...
    'MaxNumSplits',MAX_SPLITS);

%% ================================================================
% TRAIN ADABOOSTM2
% ================================================================

try

    adaModel = fitcensemble(...
        XTrain,...
        YTrain,...
        'Method','AdaBoostM2',...
        'Learners',treeTemplate,...
        'NumLearningCycles',NUM_LEARNERS,...
        'Weights',classWeights);

catch ME

    fprintf('\nWARNING: Weighted AdaBoostM2 training failed.\n');
    fprintf('Retrying without explicit class weights...\n\n');

    adaModel = fitcensemble(...
        XTrain,...
        YTrain,...
        'Method','AdaBoostM2',...
        'Learners',treeTemplate,...
        'NumLearningCycles',NUM_LEARNERS);

end

fprintf('AdaBoostM2 training completed.\n\n');

%% ================================================================
% ADABOOST PREDICTIONS
% ================================================================

[YTrainPred,trainScores] = predict(adaModel,XTrain);

[YValPred,valScores] = predict(adaModel,XVal);

[YTestPred,testScores] = predict(adaModel,XTest);

%% ================================================================
% AR(5) MODEL
% ================================================================

fprintf('============================================================\n');
fprintf('TRAINING AR(5)\n');
fprintf('============================================================\n');

% Important:
% AR(5) is trained using previous target returns.
%
% Validation and test predictions use the historical sequence
% without artificially inserting zero warm-up predictions.

[arTrainPred,arTrainActual] = arPredictions(...
    target(trainIdx),AR_ORDER);

[arValPred,arValActual] = arWalkForwardPredictions(...
    target,...
    valIdx,...
    AR_ORDER);

[arTestPred,arTestActual] = arWalkForwardPredictions(...
    target,...
    testIdx,...
    AR_ORDER);

% Refit AR(5) on train + validation for final test model
trainValIdx = [trainIdx valIdx];

arFinalModel = fitAR5(target(trainValIdx),AR_ORDER);

arFinalTestPred = zeros(length(testIdx),1);

for i=1:length(testIdx)

    absoluteIndex = testIdx(i);

    historyStart = absoluteIndex-AR_ORDER;

    historyEnd = absoluteIndex-1;

    if historyStart < 1
        arFinalTestPred(i)=NaN;
        continue;
    end

    history = target(historyStart:historyEnd);

    arFinalTestPred(i)=predictAR5FromHistory(...
        arFinalModel,...
        history);

end

arTestPred = arFinalTestPred;

arTestActual = target(testIdx);

fprintf('AR(5) training completed.\n\n');

%% ================================================================
% AR METRICS
% ================================================================

arValMetrics = regressionMetrics(arValActual,arValPred);

arTestMetrics = regressionMetrics(arTestActual,arTestPred);

arValDirectional = directionalAccuracy(...
    arValActual,...
    arValPred);

arTestDirectional = directionalAccuracy(...
    arTestActual,...
    arTestPred);

%% ================================================================
% ADABOOST CLASSIFICATION METRICS
% ================================================================

adaTrainMetrics = classificationMetrics(YTrain,YTrainPred);

adaValMetrics = classificationMetrics(YVal,YValPred);

adaTestMetrics = classificationMetrics(YTest,YTestPred);

%% ================================================================
% ADABOOST DIRECTIONAL ACCURACY
% ================================================================

adaTrainDirectional = classificationDirectionalAccuracy(...
    YTrain,...
    target(trainIdx));

adaValDirectional = classificationDirectionalAccuracy(...
    YVal,...
    target(valIdx));

adaTestDirectional = classificationDirectionalAccuracy(...
    YTest,...
    target(testIdx));

%% ================================================================
% CONVERT AR TO SIGNALS
% ================================================================

arValSignals = returnsToSignals(arValPred,THRESHOLD);

arTestSignals = returnsToSignals(arTestPred,THRESHOLD);

%% ================================================================
% ADABOOST SIGNALS
% ================================================================

adaValSignals = categoricalToSignalStrings(YValPred);

adaTestSignals = categoricalToSignalStrings(YTestPred);

%% ================================================================
% FORCE SAME LENGTH
% ================================================================

nVal = min(...
    [length(arValSignals),...
     length(adaValSignals),...
     length(target(valIdx))]);

arValSignals = arValSignals(1:nVal);

adaValSignals = adaValSignals(1:nVal);

valActualReturns = target(valIdx(1:nVal));

nTest = min(...
    [length(arTestSignals),...
     length(adaTestSignals),...
     length(target(testIdx))]);

arTestSignals = arTestSignals(1:nTest);

adaTestSignals = adaTestSignals(1:nTest);

testActualReturns = target(testIdx(1:nTest));

%% ================================================================
% BACKTEST AR(5)
% ================================================================

fprintf('============================================================\n');
fprintf('BACKTESTING AR(5)\n');
fprintf('============================================================\n');

arBacktest = backtestStrategy(...
    arTestSignals,...
    testActualReturns,...
    INITIAL_CAPITAL,...
    TRANSACTION_COST);

fprintf('Final Capital : $%.2f\n',arBacktest.finalCapital);
fprintf('Return        : %.4f%%\n',arBacktest.totalReturn*100);
fprintf('Sharpe        : %.4f\n',arBacktest.sharpe);
fprintf('Max Drawdown  : %.4f%%\n\n',arBacktest.maxDrawdown*100);

%% ================================================================
% BACKTEST ADABOOST
% ================================================================

fprintf('============================================================\n');
fprintf('BACKTESTING ADABOOSTM2\n');
fprintf('============================================================\n');

adaBacktest = backtestStrategy(...
    adaTestSignals,...
    testActualReturns,...
    INITIAL_CAPITAL,...
    TRANSACTION_COST);

fprintf('Final Capital : $%.2f\n',adaBacktest.finalCapital);
fprintf('Return        : %.4f%%\n',adaBacktest.totalReturn*100);
fprintf('Sharpe        : %.4f\n',adaBacktest.sharpe);
fprintf('Max Drawdown  : %.4f%%\n\n',adaBacktest.maxDrawdown*100);

%% ================================================================
% BUY AND HOLD
% ================================================================

buyHoldSignals = repmat("BUY",nTest,1);

buyHoldBacktest = backtestStrategy(...
    buyHoldSignals,...
    testActualReturns,...
    INITIAL_CAPITAL,...
    TRANSACTION_COST);

%% ================================================================
% PRINT MODEL RESULTS
% ================================================================

fprintf('============================================================\n');
fprintf('MODEL COMPARISON\n');
fprintf('============================================================\n\n');

fprintf('%-28s %15s %15s\n',...
    'Metric','AR(5)','AdaBoostM2');

fprintf('%s\n',repmat('-',1,62));

fprintf('%-28s %14.4f%% %14s\n',...
    'Validation RMSE',...
    arValMetrics.RMSE*100,...
    'N/A');

fprintf('%-28s %14.4f%% %14s\n',...
    'Test RMSE',...
    arTestMetrics.RMSE*100,...
    'N/A');

fprintf('%-28s %14.4f%% %14s\n',...
    'Test MAE',...
    arTestMetrics.MAE*100,...
    'N/A');

fprintf('%-28s %14.2f%% %14.2f%%\n',...
    'Directional Accuracy',...
    arTestDirectional*100,...
    adaTestDirectional*100);

fprintf('%-28s %14s %14.2f%%\n',...
    'Classification Accuracy',...
    'N/A',...
    adaTestMetrics.Accuracy*100);

fprintf('%-28s %14s %14.2f%%\n',...
    'Macro F1',...
    'N/A',...
    adaTestMetrics.MacroF1*100);

fprintf('%-28s %14s %14.2f%%\n',...
    'Balanced Accuracy',...
    'N/A',...
    adaTestMetrics.BalancedAccuracy*100);

fprintf('%-28s %14.4f%% %14.4f%%\n',...
    'Strategy Return',...
    arBacktest.totalReturn*100,...
    adaBacktest.totalReturn*100);

fprintf('%-28s %14.4f %14.4f\n',...
    'Sharpe Ratio',...
    arBacktest.sharpe,...
    adaBacktest.sharpe);

fprintf('%-28s %14.4f%% %14.4f%%\n',...
    'Maximum Drawdown',...
    arBacktest.maxDrawdown*100,...
    adaBacktest.maxDrawdown*100);

fprintf('%-28s %14.2f%% %14.2f%%\n',...
    'Win Rate',...
    arBacktest.winRate*100,...
    adaBacktest.winRate*100);

fprintf('%-28s %14d %14d\n',...
    'Number of Trades',...
    arBacktest.trades,...
    adaBacktest.trades);

fprintf('\n');

%% ================================================================
% CONFUSION MATRIX
% ================================================================

fprintf('============================================================\n');
fprintf('ADABOOST TEST CONFUSION MATRIX\n');
fprintf('============================================================\n');

confMat = confusionmat(YTest,YTestPred);

classOrder = categories(YTest);

fprintf('\n');

fprintf('%12s', '');

for i=1:length(classOrder)
    fprintf('%12s',classOrder{i});
end

fprintf('\n');

for i=1:length(classOrder)

    fprintf('%12s',classOrder{i});

    for j=1:length(classOrder)
        fprintf('%12d',confMat(i,j));
    end

    fprintf('\n');

end

fprintf('\n');

%% ================================================================
% SIGNAL COUNTS
% ================================================================

fprintf('============================================================\n');
fprintf('TEST SIGNAL DISTRIBUTION\n');
fprintf('============================================================\n');

printSignalCounts('AR(5)',arTestSignals);

printSignalCounts('AdaBoostM2',adaTestSignals);

%% ================================================================
% SAVE MAIN RESULTS TABLE
% ================================================================

Model = {
    'AR(5)';
    'AdaBoostM2'
    };

Validation_RMSE = [
    arValMetrics.RMSE;
    NaN
    ];

Test_RMSE = [
    arTestMetrics.RMSE;
    NaN
    ];

Test_MAE = [
    arTestMetrics.MAE;
    NaN
    ];

Directional_Accuracy = [
    arTestDirectional;
    adaTestDirectional
    ];

Classification_Accuracy = [
    NaN;
    adaTestMetrics.Accuracy
    ];

Macro_F1 = [
    NaN;
    adaTestMetrics.MacroF1
    ];

Balanced_Accuracy = [
    NaN;
    adaTestMetrics.BalancedAccuracy
    ];

Strategy_Return = [
    arBacktest.totalReturn;
    adaBacktest.totalReturn
    ];

Sharpe_Ratio = [
    arBacktest.sharpe;
    adaBacktest.sharpe
    ];

Max_Drawdown = [
    arBacktest.maxDrawdown;
    adaBacktest.maxDrawdown
    ];

Win_Rate = [
    arBacktest.winRate;
    adaBacktest.winRate
    ];

Trades = [
    arBacktest.trades;
    adaBacktest.trades
    ];

Final_Capital = [
    arBacktest.finalCapital;
    adaBacktest.finalCapital
    ];

comparisonTable = table(...
    Model,...
    Validation_RMSE,...
    Test_RMSE,...
    Test_MAE,...
    Directional_Accuracy,...
    Classification_Accuracy,...
    Macro_F1,...
    Balanced_Accuracy,...
    Strategy_Return,...
    Sharpe_Ratio,...
    Max_Drawdown,...
    Win_Rate,...
    Trades,...
    Final_Capital);

comparisonFile = fullfile(...
    tablesDir,...
    'ar_vs_adaboost_comparison.csv');

writetable(comparisonTable,comparisonFile);

%% ================================================================
% CLASSIFICATION METRICS TABLE
% ================================================================

Class_Model = {
    'AdaBoostM2'
    };

Accuracy = adaTestMetrics.Accuracy;

MacroF1 = adaTestMetrics.MacroF1;

BalancedAccuracy = adaTestMetrics.BalancedAccuracy;

classificationTable = table(...
    Class_Model,...
    Accuracy,...
    MacroF1,...
    BalancedAccuracy);

classificationFile = fullfile(...
    tablesDir,...
    'adaboost_classification_metrics.csv');

writetable(classificationTable,classificationFile);

%% ================================================================
% TRADING METRICS TABLE
% ================================================================

Trading_Model = {
    'AR(5)';
    'AdaBoostM2';
    'Buy & Hold'
    };

Trading_Return = [
    arBacktest.totalReturn;
    adaBacktest.totalReturn;
    buyHoldBacktest.totalReturn
    ];

Trading_Sharpe = [
    arBacktest.sharpe;
    adaBacktest.sharpe;
    buyHoldBacktest.sharpe
    ];

Trading_MaxDD = [
    arBacktest.maxDrawdown;
    adaBacktest.maxDrawdown;
    buyHoldBacktest.maxDrawdown
    ];

Trading_WinRate = [
    arBacktest.winRate;
    adaBacktest.winRate;
    buyHoldBacktest.winRate
    ];

Trading_Trades = [
    arBacktest.trades;
    adaBacktest.trades;
    buyHoldBacktest.trades
    ];

Trading_FinalCapital = [
    arBacktest.finalCapital;
    adaBacktest.finalCapital;
    buyHoldBacktest.finalCapital
    ];

tradingTable = table(...
    Trading_Model,...
    Trading_Return,...
    Trading_Sharpe,...
    Trading_MaxDD,...
    Trading_WinRate,...
    Trading_Trades,...
    Trading_FinalCapital);

tradingFile = fullfile(...
    tablesDir,...
    'ar_adaboost_trading_metrics.csv');

writetable(tradingTable,tradingFile);

%% ================================================================
% SAVE MODELS
% ================================================================

save(...
    fullfile(modelsDir,'adaboost_m2_model.mat'),...
    'adaModel',...
    'featureNames',...
    'mu',...
    'sigma',...
    'THRESHOLD',...
    'NUM_LEARNERS',...
    'MAX_SPLITS');

save(...
    fullfile(modelsDir,'ar5_model.mat'),...
    'arFinalModel',...
    'AR_ORDER',...
    'THRESHOLD');

%% ================================================================
% SAVE ALL RESULTS
% ================================================================

save(...
    fullfile(modelsDir,'ar_vs_adaboost_comparison.mat'),...
    'comparisonTable',...
    'classificationTable',...
    'tradingTable',...
    'arValMetrics',...
    'arTestMetrics',...
    'adaTrainMetrics',...
    'adaValMetrics',...
    'adaTestMetrics',...
    'arTestDirectional',...
    'adaTestDirectional',...
    'arBacktest',...
    'adaBacktest',...
    'buyHoldBacktest',...
    'confMat',...
    'arTestPred',...
    'arTestActual',...
    'YTestPred',...
    'YTest',...
    'testScores');

%% ================================================================
% FIGURE 1: PERFORMANCE COMPARISON
% ================================================================

fig1 = figure(...
    'Name','AR(5) vs AdaBoostM2 Performance',...
    'Color','w',...
    'Position',[100 100 1100 700]);

subplot(2,2,1);

values = [
    arTestDirectional*100,...
    adaTestDirectional*100
    ];

bar(values);

set(gca,...
    'XTickLabel',{'Directional Accuracy'},...
    'FontSize',10);

ylabel('Percentage');

title('Directional Accuracy');

grid on;

legend({'AR(5)','AdaBoostM2'},...
    'Location','best');

subplot(2,2,2);

values = [
    arBacktest.totalReturn*100,...
    adaBacktest.totalReturn*100
    ];

bar(values);

set(gca,...
    'XTickLabel',{'Strategy Return'},...
    'FontSize',10);

ylabel('Return (%)');

title('Trading Return');

grid on;

legend({'AR(5)','AdaBoostM2'},...
    'Location','best');

subplot(2,2,3);

values = [
    arBacktest.sharpe,...
    adaBacktest.sharpe
    ];

bar(values);

set(gca,...
    'XTickLabel',{'Sharpe Ratio'},...
    'FontSize',10);

ylabel('Sharpe');

title('Risk-Adjusted Performance');

grid on;

legend({'AR(5)','AdaBoostM2'},...
    'Location','best');

subplot(2,2,4);

values = [
    arBacktest.maxDrawdown*100,...
    adaBacktest.maxDrawdown*100
    ];

bar(values);

set(gca,...
    'XTickLabel',{'Maximum Drawdown'},...
    'FontSize',10);

ylabel('Drawdown (%)');

title('Maximum Drawdown');

grid on;

legend({'AR(5)','AdaBoostM2'},...
    'Location','best');

sgtitle('AR(5) vs Class-Balanced AdaBoostM2');

saveas(...
    fig1,...
    fullfile(figuresDir,'ar_vs_adaboost_performance.png'));

%% ================================================================
% FIGURE 2: EQUITY CURVES
% ================================================================

fig2 = figure(...
    'Name','Trading Returns Comparison',...
    'Color','w',...
    'Position',[100 100 1100 650]);

plot(...
    arBacktest.equity,...
    'LineWidth',1.8);

hold on;

plot(...
    adaBacktest.equity,...
    'LineWidth',1.8);

plot(...
    buyHoldBacktest.equity,...
    'LineWidth',1.8);

grid on;

xlabel('Test Day');

ylabel('Portfolio Value ($)');

title('AR(5) vs AdaBoostM2 vs Buy & Hold');

legend(...
    {'AR(5)','AdaBoostM2','Buy & Hold'},...
    'Location','best');

saveas(...
    fig2,...
    fullfile(figuresDir,'ar_vs_adaboost_returns.png'));

%% ================================================================
% FIGURE 3: CONFUSION MATRIX
% ================================================================

fig3 = figure(...
    'Name','AdaBoostM2 Confusion Matrix',...
    'Color','w',...
    'Position',[200 150 700 600]);

imagesc(confMat);

colorbar;

xlabel('Predicted Class');

ylabel('Actual Class');

title('AdaBoostM2 Test Confusion Matrix');

set(gca,...
    'XTick',1:length(classOrder),...
    'XTickLabel',classOrder,...
    'YTick',1:length(classOrder),...
    'YTickLabel',classOrder);

for i=1:size(confMat,1)

    for j=1:size(confMat,2)

        text(j,i,...
            num2str(confMat(i,j)),...
            'HorizontalAlignment','center',...
            'FontWeight','bold');

    end

end

saveas(...
    fig3,...
    fullfile(figuresDir,'ar_vs_adaboost_confusion.png'));

%% ================================================================
% FINAL SUMMARY
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('FINAL SUMMARY\n');
fprintf('============================================================\n\n');

fprintf('AR(5):\n');
fprintf('  Test RMSE           : %.4f%%\n',arTestMetrics.RMSE*100);
fprintf('  Test MAE            : %.4f%%\n',arTestMetrics.MAE*100);
fprintf('  Directional Accuracy: %.2f%%\n',arTestDirectional*100);
fprintf('  Strategy Return     : %.4f%%\n',arBacktest.totalReturn*100);
fprintf('  Sharpe Ratio        : %.4f\n',arBacktest.sharpe);
fprintf('  Maximum Drawdown    : %.4f%%\n',arBacktest.maxDrawdown*100);

fprintf('\nAdaBoostM2:\n');
fprintf('  Accuracy            : %.2f%%\n',adaTestMetrics.Accuracy*100);
fprintf('  Macro F1            : %.2f%%\n',adaTestMetrics.MacroF1*100);
fprintf('  Balanced Accuracy   : %.2f%%\n',adaTestMetrics.BalancedAccuracy*100);
fprintf('  Directional Accuracy: %.2f%%\n',adaTestDirectional*100);
fprintf('  Strategy Return     : %.4f%%\n',adaBacktest.totalReturn*100);
fprintf('  Sharpe Ratio        : %.4f\n',adaBacktest.sharpe);
fprintf('  Maximum Drawdown    : %.4f%%\n',adaBacktest.maxDrawdown*100);

fprintf('\nBuy & Hold:\n');
fprintf('  Strategy Return     : %.4f%%\n',buyHoldBacktest.totalReturn*100);
fprintf('  Sharpe Ratio        : %.4f\n',buyHoldBacktest.sharpe);
fprintf('  Maximum Drawdown    : %.4f%%\n',buyHoldBacktest.maxDrawdown*100);

fprintf('\n============================================================\n');
fprintf('FILES SAVED\n');
fprintf('============================================================\n');

fprintf('\nTables:\n');
fprintf('  %s\n',comparisonFile);
fprintf('  %s\n',classificationFile);
fprintf('  %s\n',tradingFile);

fprintf('\nModels:\n');
fprintf('  %s\n',...
    fullfile(modelsDir,'adaboost_m2_model.mat'));

fprintf('  %s\n',...
    fullfile(modelsDir,'ar5_model.mat'));

fprintf('\nFigures:\n');
fprintf('  %s\n',...
    fullfile(figuresDir,'ar_vs_adaboost_performance.png'));

fprintf('  %s\n',...
    fullfile(figuresDir,'ar_vs_adaboost_returns.png'));

fprintf('  %s\n',...
    fullfile(figuresDir,'ar_vs_adaboost_confusion.png'));

fprintf('\n============================================================\n');
fprintf('COMPARISON COMPLETED SUCCESSFULLY\n');
fprintf('============================================================\n\n');

end

%% =================================================================
% FIND COLUMN
% =================================================================

function name=findColumn(T,candidates)

name='';

vars=T.Properties.VariableNames;

for i=1:length(candidates)

    idx=find(...
        strcmpi(vars,candidates{i}),...
        1);

    if ~isempty(idx)

        name=vars{idx};

        return;

    end

end

end

%% =================================================================
% GET NUMERIC
% =================================================================

function x=getNumeric(T,name)

raw=T.(name);

if isnumeric(raw)

    x=double(raw);

elseif iscell(raw)

    x=str2double(string(raw));

else

    x=str2double(string(raw));

end

x=x(:);

end

%% =================================================================
% GET DATES
% =================================================================

function d=getDates(T,name)

if isempty(name)

    d=NaT(height(T),1);

    return;

end

raw=T.(name);

try

    if isdatetime(raw)

        d=raw;

    elseif isnumeric(raw)

        d=datetime(raw,'ConvertFrom','excel');

    else

        d=datetime(string(raw));

    end

catch

    d=NaT(height(T),1);

end

d=d(:);

end

%% =================================================================
% BUILD ADABOOST FEATURES
% =================================================================

function featureTable=buildAdaBoostFeatures(data,target,closePrice)

n=length(target);

% ---------------------------------------------------------------
% Find columns
% ---------------------------------------------------------------

volumeCol=findColumn(...
    data,...
    {'Volume','volume'});

sentCol=findColumn(...
    data,...
    {'Sentiment','sentiment'});

rsiCol=findColumn(...
    data,...
    {'RSI','rsi'});

% ---------------------------------------------------------------
% Volume
% ---------------------------------------------------------------

if isempty(volumeCol)

    volume=zeros(n,1);

else

    volume=getNumeric(data,volumeCol);

end

% ---------------------------------------------------------------
% Sentiment
% ---------------------------------------------------------------

if isempty(sentCol)

    sentiment=zeros(n,1);

else

    sentiment=getNumeric(data,sentCol);

end

% ---------------------------------------------------------------
% RSI
% ---------------------------------------------------------------

if isempty(rsiCol)

    RSI=calculateRSI(closePrice);

else

    RSI=getNumeric(data,rsiCol);

end

% ---------------------------------------------------------------
% Returns
% ---------------------------------------------------------------

ret1=zeros(n,1);

ret1(2:end)=...
    closePrice(2:end)./closePrice(1:end-1)-1;

ret3=zeros(n,1);

if n>3

    ret3(4:end)=...
        closePrice(4:end)./closePrice(1:end-3)-1;

end

ret5=zeros(n,1);

if n>5

    ret5(6:end)=...
        closePrice(6:end)./closePrice(1:end-5)-1;

end

ret10=zeros(n,1);

if n>10

    ret10(11:end)=...
        closePrice(11:end)./closePrice(1:end-10)-1;

end

% ---------------------------------------------------------------
% Moving averages
% ---------------------------------------------------------------

MA5=movmean(closePrice,[4 0],'omitnan');

MA10=movmean(closePrice,[9 0],'omitnan');

MA20=movmean(closePrice,[19 0],'omitnan');

MA50=movmean(closePrice,[49 0],'omitnan');

MA20Relation=closePrice./max(MA20,eps)-1;

MA50Relation=closePrice./max(MA50,eps)-1;

% ---------------------------------------------------------------
% Volatility
% ---------------------------------------------------------------

Volatility5=movstd(ret1,[4 0],'omitnan');

Volatility10=movstd(ret1,[9 0],'omitnan');

Volatility20=movstd(ret1,[19 0],'omitnan');

% ---------------------------------------------------------------
% Volume change
% ---------------------------------------------------------------

VolumeChange=zeros(n,1);

if n>1

    VolumeChange(2:end)=...
        volume(2:end)./max(volume(1:end-1),eps)-1;

end

VolumeMA20=movmean(volume,[19 0],'omitnan');

VolumeRatio=volume./max(VolumeMA20,eps);

% ---------------------------------------------------------------
% Sentiment features
% ---------------------------------------------------------------

SentimentMomentum=zeros(n,1);

if n>1

    SentimentMomentum(2:end)=...
        sentiment(2:end)-sentiment(1:end-1);

end

SentimentMA7=movmean(sentiment,[6 0],'omitnan');

SentimentMA20=movmean(sentiment,[19 0],'omitnan');

% ---------------------------------------------------------------
% Price momentum
% ---------------------------------------------------------------

MomentumMA5=closePrice./max(MA5,eps)-1;

MomentumMA10=closePrice./max(MA10,eps)-1;

MomentumMA20=closePrice./max(MA20,eps)-1;

MomentumMA50=closePrice./max(MA50,eps)-1;

% ---------------------------------------------------------------
% Trend
% ---------------------------------------------------------------

Trend5=MA5./max(MA20,eps)-1;

Trend20=MA20./max(MA50,eps)-1;

% ---------------------------------------------------------------
% Create table
% ---------------------------------------------------------------

featureTable=table(...
    ret1,...
    ret3,...
    ret5,...
    ret10,...
    RSI,...
    Volatility5,...
    Volatility10,...
    Volatility20,...
    VolumeChange,...
    VolumeRatio,...
    SentimentMomentum,...
    SentimentMA7,...
    SentimentMA20,...
    MomentumMA5,...
    MomentumMA10,...
    MomentumMA20,...
    MomentumMA50,...
    Trend5,...
    Trend20,...
    sentiment,...
    'VariableNames',{...
    'Return1D',...
    'Return3D',...
    'Return5D',...
    'Return10D',...
    'RSI',...
    'Volatility5D',...
    'Volatility10D',...
    'Volatility20D',...
    'VolumeChange',...
    'VolumeRatio',...
    'SentimentMomentum',...
    'SentimentMA7',...
    'SentimentMA20',...
    'MomentumMA5',...
    'MomentumMA10',...
    'MomentumMA20',...
    'MomentumMA50',...
    'Trend5',...
    'Trend20',...
    'Sentiment'});

end

%% =================================================================
% CALCULATE RSI
% =================================================================

function RSI=calculateRSI(price)

delta=[NaN;diff(price)];

gain=max(delta,0);

loss=max(-delta,0);

avgGain=movmean(...
    gain,...
    [13 0],...
    'omitnan');

avgLoss=movmean(...
    loss,...
    [13 0],...
    'omitnan');

RS=avgGain./max(avgLoss,eps);

RSI=100-100./(1+RS);

RSI(~isfinite(RSI))=50;

end

%% =================================================================
% FIT AR5
% =================================================================

function model=fitAR5(y,order)

y=y(:);

valid=isfinite(y);

y=y(valid);

if length(y)<=order+5

    error('Insufficient data to train AR model.');

end

N=length(y);

X=zeros(N-order,order);

Y=y(order+1:end);

for lag=1:order

    X(:,lag)=...
        y(order+1-lag:N-lag);

end

Xdesign=[ones(size(X,1),1) X];

beta=Xdesign\Y;

model.beta=beta;

model.order=order;

end

%% =================================================================
% AR TRAIN PREDICTIONS
% =================================================================

function [pred,actual]=arPredictions(y,order)

model=fitAR5(y,order);

N=length(y);

pred=zeros(N-order,1);

actual=y(order+1:end);

for i=order+1:N

    history=y(i-order:i-1);

    pred(i-order)=...
        predictAR5FromHistory(model,history);

end

end

%% =================================================================
% AR WALK FORWARD PREDICTIONS
% =================================================================

function [pred,actual]=arWalkForwardPredictions(y,idx,order)

pred=nan(length(idx),1);

actual=y(idx);

for k=1:length(idx)

    i=idx(k);

    if i<=order
        continue;
    end

    trainEnd=i-1;

    historyData=y(1:trainEnd);

    historyData=historyData(isfinite(historyData));

    if length(historyData)<=order+5
        continue;
    end

    model=fitAR5(historyData,order);

    latest=historyData(end-order+1:end);

    pred(k)=...
        predictAR5FromHistory(model,latest);

end

end

%% =================================================================
% PREDICT AR5
% =================================================================

function pred=predictAR5FromHistory(model,history)

history=history(:);

order=model.order;

if length(history)<order

    pred=NaN;

    return;

end

latest=history(end-order+1:end);

latest=flipud(latest);

X=[1 latest'];

pred=X*model.beta;

end

%% =================================================================
% REGRESSION METRICS
% =================================================================

function m=regressionMetrics(actual,pred)

valid=isfinite(actual)&isfinite(pred);

actual=actual(valid);

pred=pred(valid);

if isempty(actual)

    m.RMSE=NaN;
    m.MAE=NaN;
    m.R2=NaN;

    return;

end

err=pred-actual;

m.RMSE=sqrt(mean(err.^2));

m.MAE=mean(abs(err));

SSE=sum((actual-pred).^2);

SST=sum((actual-mean(actual)).^2);

if SST>0
    m.R2=1-SSE/SST;
else
    m.R2=NaN;
end

end

%% =================================================================
% DIRECTIONAL ACCURACY
% =================================================================

function accuracy=directionalAccuracy(actual,pred)

valid=isfinite(actual)&isfinite(pred);

actual=actual(valid);

pred=pred(valid);

if isempty(actual)

    accuracy=NaN;

    return;

end

actualDirection=sign(actual);

predDirection=sign(pred);

validDirection=...
    actualDirection~=0 &...
    predDirection~=0;

if ~any(validDirection)

    accuracy=NaN;

    return;

end

accuracy=mean(...
    actualDirection(validDirection)==...
    predDirection(validDirection));

end

%% =================================================================
% CLASSIFICATION METRICS
% =================================================================

function m=classificationMetrics(actual,pred)

actual=categorical(actual);

pred=categorical(pred);

classes=categories(actual);

cm=confusionmat(actual,pred,...
    'Order',categorical(classes));

total=sum(cm(:));

if total==0

    m.Accuracy=NaN;
    m.MacroF1=NaN;
    m.BalancedAccuracy=NaN;

    return;

end

m.Accuracy=trace(cm)/total;

numClasses=size(cm,1);

precision=zeros(numClasses,1);

recall=zeros(numClasses,1);

F1=zeros(numClasses,1);

for i=1:numClasses

    TP=cm(i,i);

    FP=sum(cm(:,i))-TP;

    FN=sum(cm(i,:))-TP;

    precision(i)=TP/max(TP+FP,eps);

    recall(i)=TP/max(TP+FN,eps);

    if precision(i)+recall(i)>0

        F1(i)=...
            2*precision(i)*recall(i)/...
            (precision(i)+recall(i));

    end

end

m.MacroF1=mean(F1);

m.BalancedAccuracy=mean(recall);

end

%% =================================================================
% CLASSIFICATION DIRECTIONAL ACCURACY
% =================================================================

function accuracy=classificationDirectionalAccuracy(...
    predictions,...
    actualReturns)

predictions=categorical(predictions);

actualReturns=actualReturns(:);

valid=isfinite(actualReturns);

predictions=predictions(valid);

actualReturns=actualReturns(valid);

predictedDirection=zeros(length(predictions),1);

predictedDirection(predictions=="BUY")=1;

predictedDirection(predictions=="SELL")=-1;

predictedDirection(predictions=="HOLD")=0;

actualDirection=zeros(length(actualReturns),1);

actualDirection(actualReturns>0)=1;

actualDirection(actualReturns<0)=-1;

% HOLD is considered directionally correct only when actual
% return is approximately neutral.

neutralZone=0.001;

actualDirection(...
    abs(actualReturns)<=neutralZone)=0;

accuracy=mean(...
    predictedDirection==actualDirection);

end

%% =================================================================
% RETURNS TO SIGNALS
% =================================================================

function signals=returnsToSignals(pred,threshold)

signals=strings(length(pred),1);

signals(:)="HOLD";

signals(pred>threshold)="BUY";

signals(pred<-threshold)="SELL";

end

%% =================================================================
% CATEGORICAL TO SIGNAL STRINGS
% =================================================================

function signals=categoricalToSignalStrings(labels)

labels=categorical(labels);

signals=strings(length(labels),1);

for i=1:length(labels)

    signals(i)=string(labels(i));

end

end

%% =================================================================
% BACKTEST STRATEGY
% =================================================================

function result=backtestStrategy(...
    signals,...
    returns,...
    initialCapital,...
    transactionCost)

signals=string(signals(:));

returns=returns(:);

n=min(length(signals),length(returns));

signals=signals(1:n);

returns=returns(1:n);

capital=initialCapital;

equity=zeros(n+1,1);

equity(1)=capital;

previousPosition=0;

dailyReturns=zeros(n,1);

tradeCount=0;

winningTrades=0;

tradeProfits=[];

for i=1:n

    switch upper(signals(i))

        case "BUY"

            position=1;

        case "SELL"

            position=-1;

        otherwise

            position=0;

    end

    % Transaction cost applied when position changes
    if position~=previousPosition

        cost=transactionCost;

        tradeCount=tradeCount+1;

    else

        cost=0;

    end

    strategyReturn=position*returns(i);

    netReturn=strategyReturn-cost;

    dailyReturns(i)=netReturn;

    capital=capital*(1+netReturn);

    equity(i+1)=capital;

    % Record profitable active periods
    if position~=0 && isfinite(netReturn)

        tradeProfits(end+1)=netReturn; %#ok<AGROW>

        if netReturn>0
            winningTrades=winningTrades+1;
        end

    end

    previousPosition=position;

end

result.equity=equity;

result.dailyReturns=dailyReturns;

result.finalCapital=capital;

result.totalReturn=capital/initialCapital-1;

%% ---------------------------------------------------------------
% Annualized return
% ---------------------------------------------------------------

if n>0

    result.annualizedReturn=...
        (capital/initialCapital)^(365/max(n,1))-1;

else

    result.annualizedReturn=NaN;

end

%% ---------------------------------------------------------------
% Sharpe
% ---------------------------------------------------------------

validReturns=dailyReturns(isfinite(dailyReturns));

if length(validReturns)>1 && std(validReturns)>0

    result.sharpe=...
        sqrt(365)*mean(validReturns)/std(validReturns);

else

    result.sharpe=NaN;

end

%% ---------------------------------------------------------------
% Sortino
% ---------------------------------------------------------------

downside=validReturns(validReturns<0);

if ~isempty(downside)

    downsideStd=sqrt(mean(downside.^2));

    if downsideStd>0

        result.sortino=...
            sqrt(365)*mean(validReturns)/downsideStd;

    else

        result.sortino=NaN;

    end

else

    result.sortino=NaN;

end

%% ---------------------------------------------------------------
% Maximum drawdown
% ---------------------------------------------------------------

runningMax=cummax(equity);

drawdown=equity./max(runningMax,eps)-1;

result.maxDrawdown=min(drawdown);

%% ---------------------------------------------------------------
% Win rate
% ---------------------------------------------------------------

if ~isempty(tradeProfits)

    result.winRate=...
        winningTrades/length(tradeProfits);

else

    result.winRate=NaN;

end

result.trades=tradeCount;

%% ---------------------------------------------------------------
% Profit factor
% ---------------------------------------------------------------

profits=tradeProfits(tradeProfits>0);

losses=tradeProfits(tradeProfits<0);

if isempty(losses)

    result.profitFactor=Inf;

elseif ~isempty(profits)

    result.profitFactor=...
        sum(profits)/abs(sum(losses));

else

    result.profitFactor=0;

end

end

%% =================================================================
% PRINT SIGNAL COUNTS
% =================================================================

function printSignalCounts(name,signals)

signals=string(signals);

buy=sum(signals=="BUY");

hold=sum(signals=="HOLD");

sell=sum(signals=="SELL");

fprintf('\n%s:\n',name);

fprintf('  BUY  : %d\n',buy);

fprintf('  HOLD : %d\n',hold);

fprintf('  SELL : %d\n',sell);

end