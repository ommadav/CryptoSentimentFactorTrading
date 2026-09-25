%% ============================================================
% STEP 4C - MULTI-MODEL CLASSIFICATION COMPARISON
% Crypto Sentiment Factor Trading
%
% Classification Models:
%   1. AdaBoostM2
%   2. RUSBoost
%   3. Random Forest
%   4. SVM-ECOC
%   5. Bagged Trees
%
% Data:
%   factor_dataset.csv
%
% Split:
%   70% Training
%   15% Validation
%   15% Test
%
% Selection:
%   Validation Macro F1
%
% IMPORTANT:
%   Test data is never used for model selection.
%% ============================================================

clear;
clc;
close all;

fprintf('\n=====================================================\n');
fprintf(' MULTI-MODEL CLASSIFICATION COMPARISON\n');
fprintf('=====================================================\n');

%% ============================================================
% PATHS
%% ============================================================

dataFile = '../data/processed/factor_dataset.csv';

resultsTablePath = ...
    '../results/tables/classification_model_comparison.csv';

resultsModelPath = ...
    '../results/models/classification_models.mat';

figurePath = ...
    '../results/figures/classification_model_comparison.png';

confusionPath = ...
    '../results/figures/classification_confusion_matrices.png';

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

fprintf('\nLoading dataset...\n');

if ~isfile(dataFile)
    error('Dataset not found: %s',dataFile);
end

T = readtable(dataFile);

fprintf('Original rows    : %d\n',height(T));
fprintf('Original columns : %d\n',width(T));

%% ============================================================
% TARGET CLEANING
%% ============================================================

if ~ismember('TargetReturn1D',T.Properties.VariableNames)

    error('TargetReturn1D column not found.');

end

targetReturn = T.TargetReturn1D;

validTarget = isfinite(targetReturn);

T = T(validTarget,:);

targetReturn = T.TargetReturn1D;

fprintf('Rows after target cleaning : %d\n',height(T));

%% ============================================================
% CREATE BUY / HOLD / SELL TARGET
%% ============================================================

fprintf('\nCreating BUY / HOLD / SELL classes...\n');

% 0.9% classification threshold

threshold = 0.009;

targetClass = strings(height(T),1);

targetClass(targetReturn > threshold) = "BUY";

targetClass(targetReturn < -threshold) = "SELL";

targetClass( ...
    targetReturn >= -threshold & ...
    targetReturn <= threshold) = "HOLD";

Y = categorical( ...
    targetClass,...
    ["SELL","HOLD","BUY"]);

%% ============================================================
% CLASS DISTRIBUTION
%% ============================================================

fprintf('\nClass distribution:\n');

classNames = ["SELL","HOLD","BUY"];

for i = 1:numel(classNames)

    classCount = sum(Y == classNames(i));

    fprintf( ...
        '%-6s : %4d (%.2f%%)\n',...
        classNames(i),...
        classCount,...
        100*classCount/numel(Y));

end

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

%% Keep only numeric features

numericMask = false(size(featureNames));

for i = 1:numel(featureNames)

    numericMask(i) = ...
        isnumeric(T.(featureNames{i}));

end

featureNames = featureNames(numericMask);

fprintf( ...
    'Number of candidate features : %d\n',...
    numel(featureNames));

%% ============================================================
% CREATE FEATURE MATRIX
%% ============================================================

X = T{:,featureNames};

%% ============================================================
% REMOVE INVALID FEATURE ROWS
%% ============================================================

validFeatures = all(isfinite(X),2);

fprintf( ...
    'Rows with valid features : %d\n',...
    sum(validFeatures));

fprintf( ...
    'Rows removed due to feature NaN/Inf : %d\n',...
    sum(~validFeatures));

X = X(validFeatures,:);

Y = Y(validFeatures);

T = T(validFeatures,:);

fprintf('\nFinal usable dataset:\n');

fprintf('Rows     : %d\n',size(X,1));

fprintf('Features : %d\n',size(X,2));

%% ============================================================
% DISPLAY FEATURES
%% ============================================================

fprintf('\nFeatures used:\n');

for i = 1:numel(featureNames)

    fprintf( ...
        '%4d. %s\n',...
        i,...
        featureNames{i});

end

%% ============================================================
% TIME-ORDERED SPLIT
%% ============================================================

n = size(X,1);

nTrain = floor(0.70*n);

nVal = floor(0.15*n);

nTest = n - nTrain - nVal;

idxTrain = 1:nTrain;

idxVal = ...
    nTrain+1:nTrain+nVal;

idxTest = ...
    nTrain+nVal+1:n;

Xtrain = X(idxTrain,:);

Ytrain = Y(idxTrain);

Xval = X(idxVal,:);

Yval = Y(idxVal);

Xtest = X(idxTest,:);

Ytest = Y(idxTest);

fprintf('\n=====================================================\n');
fprintf(' DATA SPLIT\n');
fprintf('=====================================================\n');

fprintf('Training   : %d observations\n',nTrain);

fprintf('Validation : %d observations\n',nVal);

fprintf('Testing    : %d observations\n',nTest);

%% ============================================================
% DATE RANGES
%% ============================================================

if ismember('Date',T.Properties.VariableNames)

    try

        fprintf('\nDate ranges:\n');

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

        fprintf('\nDate ranges unavailable.\n');

    end

end

%% ============================================================
% STANDARDIZATION FOR SVM
%
% IMPORTANT:
% Training statistics only.
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

numModels = 5;

ModelNames = strings(numModels,1);

ValAccuracy = zeros(numModels,1);

ValMacroF1 = zeros(numModels,1);

ValBalancedAccuracy = zeros(numModels,1);

TestAccuracy = zeros(numModels,1);

TestMacroF1 = zeros(numModels,1);

TestBalancedAccuracy = zeros(numModels,1);

Models = struct();

%% ============================================================
% MODEL 1 - ADABOOST M2
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 1: ADABOOST M2\n');
fprintf('=====================================================\n');

fprintf('Training AdaBoostM2...\n');

AdaModel = fitcensemble( ...
    Xtrain,...
    Ytrain,...
    'Method','AdaBoostM2',...
    'NumLearningCycles',200);

YPredVal_Ada = ...
    predict(AdaModel,Xval);

[valAcc,valF1,valBal] = ...
    evaluateModel(Yval,YPredVal_Ada);

YPredTest_Ada = ...
    predict(AdaModel,Xtest);

[testAcc,testF1,testBal] = ...
    evaluateModel(Ytest,YPredTest_Ada);

fprintf('\nValidation:\n');

fprintf('Accuracy       : %.2f %%\n',100*valAcc);

fprintf('Macro F1       : %.2f %%\n',100*valF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*valBal);

fprintf('\nTest:\n');

fprintf('Accuracy       : %.2f %%\n',100*testAcc);

fprintf('Macro F1       : %.2f %%\n',100*testF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*testBal);

ModelNames(1) = "AdaBoostM2";

ValAccuracy(1) = valAcc;

ValMacroF1(1) = valF1;

ValBalancedAccuracy(1) = valBal;

TestAccuracy(1) = testAcc;

TestMacroF1(1) = testF1;

TestBalancedAccuracy(1) = testBal;

Models.AdaBoostM2 = AdaModel;

%% ============================================================
% MODEL 2 - RUSBOOST
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 2: RUSBOOST\n');
fprintf('=====================================================\n');

fprintf('Training RUSBoost...\n');

RusModel = fitcensemble( ...
    Xtrain,...
    Ytrain,...
    'Method','RUSBoost',...
    'NumLearningCycles',200);

YPredVal_Rus = ...
    predict(RusModel,Xval);

[valAcc,valF1,valBal] = ...
    evaluateModel(Yval,YPredVal_Rus);

YPredTest_Rus = ...
    predict(RusModel,Xtest);

[testAcc,testF1,testBal] = ...
    evaluateModel(Ytest,YPredTest_Rus);

fprintf('\nValidation:\n');

fprintf('Accuracy       : %.2f %%\n',100*valAcc);

fprintf('Macro F1       : %.2f %%\n',100*valF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*valBal);

fprintf('\nTest:\n');

fprintf('Accuracy       : %.2f %%\n',100*testAcc);

fprintf('Macro F1       : %.2f %%\n',100*testF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*testBal);

ModelNames(2) = "RUSBoost";

ValAccuracy(2) = valAcc;

ValMacroF1(2) = valF1;

ValBalancedAccuracy(2) = valBal;

TestAccuracy(2) = testAcc;

TestMacroF1(2) = testF1;

TestBalancedAccuracy(2) = testBal;

Models.RUSBoost = RusModel;

%% ============================================================
% MODEL 3 - RANDOM FOREST
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 3: RANDOM FOREST\n');
fprintf('=====================================================\n');

fprintf('Training Random Forest...\n');

RFModel = TreeBagger( ...
    300,...
    Xtrain,...
    Ytrain,...
    'Method','classification',...
    'OOBPrediction','On');

YPredVal_RF = ...
    predict(RFModel,Xval);

YPredVal_RF = ...
    categorical(YPredVal_RF);

[valAcc,valF1,valBal] = ...
    evaluateModel(Yval,YPredVal_RF);

YPredTest_RF = ...
    predict(RFModel,Xtest);

YPredTest_RF = ...
    categorical(YPredTest_RF);

[testAcc,testF1,testBal] = ...
    evaluateModel(Ytest,YPredTest_RF);

fprintf('\nValidation:\n');

fprintf('Accuracy       : %.2f %%\n',100*valAcc);

fprintf('Macro F1       : %.2f %%\n',100*valF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*valBal);

fprintf('\nTest:\n');

fprintf('Accuracy       : %.2f %%\n',100*testAcc);

fprintf('Macro F1       : %.2f %%\n',100*testF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*testBal);

ModelNames(3) = "RandomForest";

ValAccuracy(3) = valAcc;

ValMacroF1(3) = valF1;

ValBalancedAccuracy(3) = valBal;

TestAccuracy(3) = testAcc;

TestMacroF1(3) = testF1;

TestBalancedAccuracy(3) = testBal;

Models.RandomForest = RFModel;

%% ============================================================
% MODEL 4 - SVM ECOC
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 4: SVM-ECOC\n');
fprintf('=====================================================\n');

fprintf('Training SVM-ECOC...\n');

SVMModel = fitcecoc( ...
    XtrainSVM,...
    Ytrain,...
    'Learners','linear');

YPredVal_SVM = ...
    predict(SVMModel,XvalSVM);

[valAcc,valF1,valBal] = ...
    evaluateModel(Yval,YPredVal_SVM);

YPredTest_SVM = ...
    predict(SVMModel,XtestSVM);

[testAcc,testF1,testBal] = ...
    evaluateModel(Ytest,YPredTest_SVM);

fprintf('\nValidation:\n');

fprintf('Accuracy       : %.2f %%\n',100*valAcc);

fprintf('Macro F1       : %.2f %%\n',100*valF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*valBal);

fprintf('\nTest:\n');

fprintf('Accuracy       : %.2f %%\n',100*testAcc);

fprintf('Macro F1       : %.2f %%\n',100*testF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*testBal);

ModelNames(4) = "SVM-ECOC";

ValAccuracy(4) = valAcc;

ValMacroF1(4) = valF1;

ValBalancedAccuracy(4) = valBal;

TestAccuracy(4) = testAcc;

TestMacroF1(4) = testF1;

TestBalancedAccuracy(4) = testBal;

Models.SVM_ECOC = SVMModel;

%% ============================================================
% MODEL 5 - BAGGED TREES
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' MODEL 5: BAGGED TREES\n');
fprintf('=====================================================\n');

fprintf('Training Bagged Trees...\n');

BagModel = fitcensemble( ...
    Xtrain,...
    Ytrain,...
    'Method','Bag',...
    'NumLearningCycles',300);

YPredVal_Bag = ...
    predict(BagModel,Xval);

[valAcc,valF1,valBal] = ...
    evaluateModel(Yval,YPredVal_Bag);

YPredTest_Bag = ...
    predict(BagModel,Xtest);

[testAcc,testF1,testBal] = ...
    evaluateModel(Ytest,YPredTest_Bag);

fprintf('\nValidation:\n');

fprintf('Accuracy       : %.2f %%\n',100*valAcc);

fprintf('Macro F1       : %.2f %%\n',100*valF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*valBal);

fprintf('\nTest:\n');

fprintf('Accuracy       : %.2f %%\n',100*testAcc);

fprintf('Macro F1       : %.2f %%\n',100*testF1);

fprintf('Balanced Acc.  : %.2f %%\n',100*testBal);

ModelNames(5) = "BaggedTrees";

ValAccuracy(5) = valAcc;

ValMacroF1(5) = valF1;

ValBalancedAccuracy(5) = valBal;

TestAccuracy(5) = testAcc;

TestMacroF1(5) = testF1;

TestBalancedAccuracy(5) = testBal;

Models.BaggedTrees = BagModel;

%% ============================================================
% RESULTS TABLE
%% ============================================================

Results = table( ...
    ModelNames,...
    ValAccuracy * 100,...
    ValMacroF1 * 100,...
    ValBalancedAccuracy * 100,...
    TestAccuracy * 100,...
    TestMacroF1 * 100,...
    TestBalancedAccuracy * 100,...
    'VariableNames',{ ...
    'Model',...
    'ValidationAccuracyPct',...
    'ValidationMacroF1Pct',...
    'ValidationBalancedAccuracyPct',...
    'TestAccuracyPct',...
    'TestMacroF1Pct',...
    'TestBalancedAccuracyPct'});

%% ============================================================
% DISPLAY FINAL RESULTS
%% ============================================================

fprintf('\n\n=====================================================\n');
fprintf(' FINAL MODEL COMPARISON\n');
fprintf('=====================================================\n\n');

disp(Results);

%% ============================================================
% MODEL SELECTION
%
% ONLY VALIDATION DATA IS USED.
%% ============================================================

[~,bestIdx] = ...
    max(Results.ValidationMacroF1Pct);

bestModel = Results.Model(bestIdx);

fprintf('\n=====================================================\n');
fprintf(' MODEL SELECTED USING VALIDATION MACRO F1\n');
fprintf('=====================================================\n');

fprintf('Selected Model : %s\n',bestModel);

fprintf('\nValidation performance:\n');

fprintf('Accuracy       : %.2f %%\n',...
    Results.ValidationAccuracyPct(bestIdx));

fprintf('Macro F1       : %.2f %%\n',...
    Results.ValidationMacroF1Pct(bestIdx));

fprintf('Balanced Acc.  : %.2f %%\n',...
    Results.ValidationBalancedAccuracyPct(bestIdx));

fprintf('\nUntouched test performance:\n');

fprintf('Accuracy       : %.2f %%\n',...
    Results.TestAccuracyPct(bestIdx));

fprintf('Macro F1       : %.2f %%\n',...
    Results.TestMacroF1Pct(bestIdx));

fprintf('Balanced Acc.  : %.2f %%\n',...
    Results.TestBalancedAccuracyPct(bestIdx));

%% ============================================================
% SORTED RESULTS
%% ============================================================

SortedResults = ...
    sortrows( ...
        Results,...
        'ValidationMacroF1Pct',...
        'descend');

fprintf('\n=====================================================\n');
fprintf(' MODELS SORTED BY VALIDATION MACRO F1\n');
fprintf('=====================================================\n\n');

disp(SortedResults);

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
    'SortedResults',...
    'featureNames',...
    'mu',...
    'sigma',...
    'threshold',...
    'bestModel');

fprintf('\nResults saved:\n');

fprintf('  %s\n',resultsTablePath);

fprintf('  %s\n',resultsModelPath);

%% ============================================================
% PERFORMANCE GRAPH
%% ============================================================

figure( ...
    'Name','Classification Model Comparison',...
    'Color','w');

metrics = [ ...
    Results.TestAccuracyPct,...
    Results.TestMacroF1Pct,...
    Results.TestBalancedAccuracyPct];

bar(metrics);

grid on;

xlabel('Model');

ylabel('Performance (%)');

title('Test-Set Classification Model Comparison');

set(gca,...
    'XTick',1:height(Results),...
    'XTickLabel',Results.Model);

xtickangle(25);

legend( ...
    {'Accuracy','Macro F1','Balanced Accuracy'},...
    'Location','best');

saveas(gcf,figurePath);

fprintf('  %s\n',figurePath);

%% ============================================================
% CONFUSION MATRICES
%% ============================================================

figure( ...
    'Name','Test Confusion Matrices',...
    'Color','w');

tiledlayout(2,3);

%% AdaBoostM2

nexttile;

confusionchart( ...
    Ytest,...
    YPredTest_Ada);

title('AdaBoostM2');

%% RUSBoost

nexttile;

confusionchart( ...
    Ytest,...
    YPredTest_Rus);

title('RUSBoost');

%% Random Forest

nexttile;

confusionchart( ...
    Ytest,...
    YPredTest_RF);

title('Random Forest');

%% SVM

nexttile;

confusionchart( ...
    Ytest,...
    YPredTest_SVM);

title('SVM-ECOC');

%% Bagged Trees

nexttile;

confusionchart( ...
    Ytest,...
    YPredTest_Bag);

title('Bagged Trees');

saveas(gcf,confusionPath);

fprintf('  %s\n',confusionPath);

%% ============================================================
% FINAL SUMMARY
%% ============================================================

fprintf('\n=====================================================\n');
fprintf(' EXPERIMENT COMPLETED SUCCESSFULLY\n');
fprintf('=====================================================\n');

fprintf('\nSelected model using validation Macro F1: %s\n',...
    bestModel);

fprintf('\nComplete results:\n');

for i = 1:height(Results)

    fprintf('\n%s\n',Results.Model(i));

    fprintf( ...
        '  Validation Accuracy       : %.2f %%\n',...
        Results.ValidationAccuracyPct(i));

    fprintf( ...
        '  Validation Macro F1       : %.2f %%\n',...
        Results.ValidationMacroF1Pct(i));

    fprintf( ...
        '  Validation Balanced Acc.  : %.2f %%\n',...
        Results.ValidationBalancedAccuracyPct(i));

    fprintf( ...
        '  Test Accuracy             : %.2f %%\n',...
        Results.TestAccuracyPct(i));

    fprintf( ...
        '  Test Macro F1             : %.2f %%\n',...
        Results.TestMacroF1Pct(i));

    fprintf( ...
        '  Test Balanced Accuracy    : %.2f %%\n',...
        Results.TestBalancedAccuracyPct(i));

end

fprintf('\n=====================================================\n');


%% ============================================================
% LOCAL FUNCTION
% EVALUATE CLASSIFICATION MODEL
%% ============================================================

function [accuracy,macroF1,balancedAccuracy] = ...
    evaluateModel(Ytrue,Ypred)

    %% Convert to categorical

    if ~iscategorical(Ytrue)

        Ytrue = categorical(Ytrue);

    end

    if ~iscategorical(Ypred)

        Ypred = categorical(Ypred);

    end

    %% Force common class names

    classNames = ["SELL","HOLD","BUY"];

    Ytrue = categorical( ...
        string(Ytrue),...
        classNames);

    Ypred = categorical( ...
        string(Ypred),...
        classNames);

    %% Accuracy

    accuracy = mean(Ytrue == Ypred);

    %% Confusion matrix

    classOrder = categorical( ...
        classNames,...
        classNames);

    C = confusionmat( ...
        Ytrue,...
        Ypred,...
        'Order',classOrder);

    %% Per-class metrics

    nClasses = numel(classNames);

    precision = zeros(nClasses,1);

    recall = zeros(nClasses,1);

    F1 = zeros(nClasses,1);

    for k = 1:nClasses

        TP = C(k,k);

        FP = sum(C(:,k)) - TP;

        FN = sum(C(k,:)) - TP;

        %% Precision

        if TP + FP > 0

            precision(k) = ...
                TP/(TP + FP);

        else

            precision(k) = 0;

        end

        %% Recall

        if TP + FN > 0

            recall(k) = ...
                TP/(TP + FN);

        else

            recall(k) = 0;

        end

        %% F1

        if precision(k) + recall(k) > 0

            F1(k) = ...
                2 * precision(k) * recall(k) / ...
                (precision(k) + recall(k));

        else

            F1(k) = 0;

        end

    end

    %% Macro F1

    macroF1 = mean(F1);

    %% Balanced Accuracy

    balancedAccuracy = mean(recall);

end