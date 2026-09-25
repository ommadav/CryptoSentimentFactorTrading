function step4_model_comparison()
%STEP4_MODEL_COMPARISON
% Compare models for next-day BTC return prediction.
%
% Models:
%   1. Linear Regression
%   2. Ridge Regression
%   3. Bagged Trees
%   4. Support Vector Regression
%   5. LSTM
%   6. Autoregressive AR(5)
%
% Input:
%   data/processed/factor_dataset.csv
%
% Outputs:
%   results/models/model_comparison.mat
%   results/tables/model_comparison.csv
%   results/figures/model_comparison.png
%
% Leakage protection:
%   - Chronological train/validation/test split
%   - Missing-value imputation from training data only
%   - Feature scaling from training data only
%   - Model selection based ONLY on validation performance
%
% NOTE:
% LSTM is optional. If Deep Learning Toolbox is unavailable,
% it is skipped.

clc;
close all;

fprintf('\n============================================================\n');
fprintf(' [STEP 4] MODEL COMPARISON\n');
fprintf('============================================================\n\n');

%% ============================================================
% 1. PROJECT PATHS
% ============================================================

thisFile = mfilename('fullpath');

if isempty(thisFile)
    srcDir = pwd;
else
    srcDir = fileparts(thisFile);
end

projectDir = fileparts(srcDir);

addpath(srcDir);
addpath(fullfile(srcDir, 'utils'));

fprintf('Project directory:\n%s\n\n', projectDir);

%% ============================================================
% 2. PATHS
% ============================================================

inputFile = fullfile( ...
    projectDir, ...
    'data', ...
    'processed', ...
    'factor_dataset.csv');

modelDir = fullfile( ...
    projectDir, ...
    'results', ...
    'models');

tableDir = fullfile( ...
    projectDir, ...
    'results', ...
    'tables');

figureDir = fullfile( ...
    projectDir, ...
    'results', ...
    'figures');

if ~exist(modelDir, 'dir')
    mkdir(modelDir);
end

if ~exist(tableDir, 'dir')
    mkdir(tableDir);
end

if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

%% ============================================================
% 3. CHECK INPUT
% ============================================================

if ~isfile(inputFile)

    error('step4:missingInput', ...
        ['Factor dataset not found:\n%s\n\n' ...
         'Run Step 3 first.'], ...
        inputFile);

end

%% ============================================================
% 4. LOAD DATA
% ============================================================

fprintf('-> Loading factor dataset...\n');
fprintf('   File: %s\n\n', inputFile);

try

    data = readtable( ...
        inputFile, ...
        'VariableNamingRule', 'preserve');

catch

    try

        data = readtable( ...
            inputFile, ...
            'PreserveVariableNames', true);

    catch ME

        error('step4:readError', ...
            'Unable to read factor dataset.\n%s', ...
            ME.message);

    end

end

if isempty(data) || height(data) == 0

    error('step4:emptyData', ...
        'Factor dataset is empty.');

end

%% ============================================================
% 5. VALIDATE TARGET
% ============================================================

if ~ismember( ...
        'TargetReturn1D', ...
        data.Properties.VariableNames)

    error('step4:missingTarget', ...
        'TargetReturn1D is missing.');

end

if ~ismember( ...
        'Date', ...
        data.Properties.VariableNames)

    error('step4:missingDate', ...
        'Date column is missing.');

end

%% ============================================================
% 6. DATE CONVERSION
% ============================================================

if ~isdatetime(data.Date)

    try

        data.Date = datetime(data.Date);

    catch ME

        error('step4:dateError', ...
            'Unable to convert Date column.\n%s', ...
            ME.message);

    end

end

data = sortrows(data, 'Date');

%% ============================================================
% 7. REMOVE INVALID TARGET ROWS
% ============================================================

validRows = ...
    ~isnat(data.Date) & ...
    isfinite(data.TargetReturn1D);

data = data(validRows,:);

if height(data) < 100

    error('step4:insufficientData', ...
        'Too few observations remain.');

end

fprintf('Valid modeling observations: %d\n\n', ...
    height(data));

%% ============================================================
% 8. CHRONOLOGICAL SPLIT
% ============================================================

n = height(data);

trainEnd = floor(0.70 * n);

valEnd = floor(0.85 * n);

trainIdx = (1:trainEnd)';

valIdx = (trainEnd+1:valEnd)';

testIdx = (valEnd+1:n)';

fprintf('------------------------------------------------------------\n');
fprintf(' [DATA SPLIT]\n');
fprintf('------------------------------------------------------------\n');

fprintf('Total observations : %d\n', n);

fprintf('Training           : %d\n', numel(trainIdx));

fprintf('Validation         : %d\n', numel(valIdx));

fprintf('Test               : %d\n', numel(testIdx));

fprintf('\n');

fprintf('Training dates:\n');

fprintf('  %s -> %s\n', ...
    datestr(data.Date(trainIdx(1)), 'yyyy-mm-dd'), ...
    datestr(data.Date(trainIdx(end)), 'yyyy-mm-dd'));

fprintf('Validation dates:\n');

fprintf('  %s -> %s\n', ...
    datestr(data.Date(valIdx(1)), 'yyyy-mm-dd'), ...
    datestr(data.Date(valIdx(end)), 'yyyy-mm-dd'));

fprintf('Test dates:\n');

fprintf('  %s -> %s\n', ...
    datestr(data.Date(testIdx(1)), 'yyyy-mm-dd'), ...
    datestr(data.Date(testIdx(end)), 'yyyy-mm-dd'));

fprintf('------------------------------------------------------------\n\n');

%% ============================================================
% 9. FIND NUMERIC FEATURES
% ============================================================

excluded = { ...
    'Date', ...
    'TargetReturn1D', ...
    'TargetDirection', ...
    'SentimentClass'};

candidateFeatures = {};

for k = 1:width(data)

    name = data.Properties.VariableNames{k};

    if ismember(name, excluded)

        continue;

    end

    column = data.(name);

    if isnumeric(column)

        candidateFeatures{end+1} = name; %#ok<AGROW>

    end

end

if isempty(candidateFeatures)

    error('step4:noFeatures', ...
        'No numeric predictor features found.');

end

fprintf('Candidate numeric features: %d\n', ...
    numel(candidateFeatures));

%% ============================================================
% 10. REMOVE CONSTANT / INVALID FEATURES
% ============================================================

keepFeatures = {};

for k = 1:numel(candidateFeatures)

    x = data.(candidateFeatures{k});

    trainX = x(trainIdx);

    valid = isfinite(trainX);

    if sum(valid) < 10

        continue;

    end

    s = std(trainX(valid), 0);

    if isfinite(s) && s > 1e-12

        keepFeatures{end+1} = ...
            candidateFeatures{k}; %#ok<AGROW>

    end

end

candidateFeatures = keepFeatures;

fprintf('Usable numeric features: %d\n\n', ...
    numel(candidateFeatures));

%% ============================================================
% 11. BUILD MATRIX
% ============================================================

p = numel(candidateFeatures);

X = nan(n,p);

for j = 1:p

    X(:,j) = data.(candidateFeatures{j});

end

y = data.TargetReturn1D(:);

%% ============================================================
% 12. TRAINING-ONLY MEDIAN IMPUTATION
% ============================================================

featureMedian = nan(1,p);

for j = 1:p

    values = X(trainIdx,j);

    values = values(isfinite(values));

    if isempty(values)

        featureMedian(j) = 0;

    else

        featureMedian(j) = median(values);

    end

end

for j = 1:p

    missing = ~isfinite(X(:,j));

    X(missing,j) = featureMedian(j);

end

%% ============================================================
% 13. TRAINING-ONLY STANDARDIZATION
% ============================================================

mu = mean( ...
    X(trainIdx,:), ...
    1);

sigma = std( ...
    X(trainIdx,:), ...
    0, ...
    1);

sigma(~isfinite(sigma) | sigma < 1e-12) = 1;

XScaled = ...
    (X - mu) ./ sigma;

%% ============================================================
% 14. SPLIT MATRICES
% ============================================================

XTrain = XScaled(trainIdx,:);

XVal = XScaled(valIdx,:);

XTest = XScaled(testIdx,:);

yTrain = y(trainIdx);

yVal = y(valIdx);

yTest = y(testIdx);

%% ============================================================
% 15. MODEL STORAGE
% ============================================================

modelNames = { ...
    'Linear Regression', ...
    'Ridge Regression', ...
    'Bagged Trees', ...
    'SVR', ...
    'LSTM', ...
    'AR(5)'};

numModels = numel(modelNames);

valRMSE = nan(numModels,1);

valMAE = nan(numModels,1);

valR2 = nan(numModels,1);

testRMSE = nan(numModels,1);

testMAE = nan(numModels,1);

testR2 = nan(numModels,1);

valPredictions = cell(numModels,1);

testPredictions = cell(numModels,1);

trainedModels = cell(numModels,1);

%% ============================================================
% MODEL 1: LINEAR REGRESSION
% ============================================================

fprintf('\n============================================================\n');
fprintf(' MODEL 1: LINEAR REGRESSION\n');
fprintf('============================================================\n');

try

    % Use pseudoinverse / least squares rather than fitlm.
    %
    % This avoids rank-deficiency warnings caused by highly
    % correlated technical indicators.

    XTrainOLS = [ones(size(XTrain,1),1), XTrain];

    beta = pinv(XTrainOLS) * yTrain;

    XValOLS = [ones(size(XVal,1),1), XVal];

    XTestOLS = [ones(size(XTest,1),1), XTest];

    predVal = XValOLS * beta;

    predTest = XTestOLS * beta;

    trainedModels{1} = beta;

    valPredictions{1} = predVal;

    testPredictions{1} = predTest;

    [valRMSE(1),valMAE(1),valR2(1)] = ...
        calculateMetrics(yVal,predVal);

    [testRMSE(1),testMAE(1),testR2(1)] = ...
        calculateMetrics(yTest,predTest);

    printMetrics( ...
        valRMSE(1), ...
        valMAE(1), ...
        valR2(1), ...
        testRMSE(1), ...
        testMAE(1), ...
        testR2(1));

catch ME

    warning('step4:linearFailed', ...
        'Linear regression failed: %s', ME.message);

end

%% ============================================================
% MODEL 2: RIDGE REGRESSION
% ============================================================

fprintf('\n============================================================\n');
fprintf(' MODEL 2: RIDGE REGRESSION\n');
fprintf('============================================================\n');

try

    lambda = 1;

    ridgeModel = fitrlinear( ...
        XTrain, ...
        yTrain, ...
        'Learner', 'leastsquares', ...
        'Regularization', 'ridge', ...
        'Lambda', lambda);

    predVal = predict( ...
        ridgeModel, ...
        XVal);

    predTest = predict( ...
        ridgeModel, ...
        XTest);

    trainedModels{2} = ridgeModel;

    valPredictions{2} = predVal;

    testPredictions{2} = predTest;

    [valRMSE(2),valMAE(2),valR2(2)] = ...
        calculateMetrics(yVal,predVal);

    [testRMSE(2),testMAE(2),testR2(2)] = ...
        calculateMetrics(yTest,predTest);

    fprintf('Lambda: %.4f\n',lambda);

    printMetrics( ...
        valRMSE(2), ...
        valMAE(2), ...
        valR2(2), ...
        testRMSE(2), ...
        testMAE(2), ...
        testR2(2));

catch ME

    warning('step4:ridgeFailed', ...
        'Ridge regression failed: %s', ME.message);

end

%% ============================================================
% MODEL 3: BAGGED TREES
% ============================================================

fprintf('\n============================================================\n');
fprintf(' MODEL 3: BAGGED TREES\n');
fprintf('============================================================\n');

try

    treeTemplate = templateTree( ...
        'MaxNumSplits',20);

    numTrees = 100;

    baggedModel = fitrensemble( ...
        XTrain, ...
        yTrain, ...
        'Method','Bag', ...
        'NumLearningCycles',numTrees, ...
        'Learners',treeTemplate);

    predVal = predict( ...
        baggedModel, ...
        XVal);

    predTest = predict( ...
        baggedModel, ...
        XTest);

    trainedModels{3} = baggedModel;

    valPredictions{3} = predVal;

    testPredictions{3} = predTest;

    [valRMSE(3),valMAE(3),valR2(3)] = ...
        calculateMetrics(yVal,predVal);

    [testRMSE(3),testMAE(3),testR2(3)] = ...
        calculateMetrics(yTest,predTest);

    fprintf('Number of trees: %d\n',numTrees);

    printMetrics( ...
        valRMSE(3), ...
        valMAE(3), ...
        valR2(3), ...
        testRMSE(3), ...
        testMAE(3), ...
        testR2(3));

catch ME

    warning('step4:baggedFailed', ...
        'Bagged trees failed: %s',ME.message);

end

%% ============================================================
% MODEL 4: SUPPORT VECTOR REGRESSION
% ============================================================

fprintf('\n============================================================\n');
fprintf(' MODEL 4: SUPPORT VECTOR REGRESSION\n');
fprintf('============================================================\n');

try

    svrModel = fitrsvm( ...
        XTrain, ...
        yTrain, ...
        'KernelFunction','gaussian', ...
        'KernelScale','auto', ...
        'Standardize',false);

    predVal = predict( ...
        svrModel, ...
        XVal);

    predTest = predict( ...
        svrModel, ...
        XTest);

    trainedModels{4} = svrModel;

    valPredictions{4} = predVal;

    testPredictions{4} = predTest;

    [valRMSE(4),valMAE(4),valR2(4)] = ...
        calculateMetrics(yVal,predVal);

    [testRMSE(4),testMAE(4),testR2(4)] = ...
        calculateMetrics(yTest,predTest);

    printMetrics( ...
        valRMSE(4), ...
        valMAE(4), ...
        valR2(4), ...
        testRMSE(4), ...
        testMAE(4), ...
        testR2(4));

catch ME

    warning('step4:svrFailed', ...
        'SVR failed: %s',ME.message);

end

%% ============================================================
% MODEL 5: LSTM
% ============================================================

fprintf('\n============================================================\n');
fprintf(' MODEL 5: LSTM\n');
fprintf('============================================================\n');

lstmAvailable = ...
    exist('trainNetwork','file') == 2 && ...
    exist('sequenceInputLayer','file') == 2;

if lstmAvailable

    try

        % -----------------------------------------------------
        % Use a sequence of observations.
        %
        % Each timestep contains all engineered features.
        % -----------------------------------------------------

        XTrainSeq = XTrain';

        YTrainSeq = yTrain';

        layers = [ ...
            sequenceInputLayer(p)
            lstmLayer(32,'OutputMode','sequence')
            dropoutLayer(0.10)
            fullyConnectedLayer(1)
            regressionLayer];

        options = trainingOptions( ...
            'adam', ...
            'MaxEpochs',30, ...
            'MiniBatchSize',32, ...
            'InitialLearnRate',0.0005, ...
            'GradientThreshold',1, ...
            'Shuffle','never', ...
            'Verbose',false);

        fprintf('Training LSTM...\n');

        lstmModel = trainNetwork( ...
            {XTrainSeq}, ...
            {YTrainSeq}, ...
            layers, ...
            options);

        % Validation
        valOutput = predict( ...
            lstmModel, ...
            {XVal'}, ...
            'MiniBatchSize',1);

        predVal = extractSequencePrediction( ...
            valOutput, ...
            numel(yVal));

        % Test
        testOutput = predict( ...
            lstmModel, ...
            {XTest'}, ...
            'MiniBatchSize',1);

        predTest = extractSequencePrediction( ...
            testOutput, ...
            numel(yTest));

        trainedModels{5} = lstmModel;

        valPredictions{5} = predVal;

        testPredictions{5} = predTest;

        [valRMSE(5),valMAE(5),valR2(5)] = ...
            calculateMetrics(yVal,predVal);

        [testRMSE(5),testMAE(5),testR2(5)] = ...
            calculateMetrics(yTest,predTest);

        printMetrics( ...
            valRMSE(5), ...
            valMAE(5), ...
            valR2(5), ...
            testRMSE(5), ...
            testMAE(5), ...
            testR2(5));

    catch ME

        warning('step4:lstmFailed', ...
            ['LSTM failed and will be skipped.\n' ...
             'Reason: %s'],ME.message);

    end

else

    fprintf('Deep Learning Toolbox not available.\n');

    fprintf('LSTM will be skipped.\n');

end

%% ============================================================
% MODEL 6: AR(5)
% ============================================================

fprintf('\n============================================================\n');
fprintf(' MODEL 6: AUTOREGRESSIVE AR(5)\n');
fprintf('============================================================\n');

try

    lagOrder = 5;

    returnSeries = data.Return;

    if ~isnumeric(returnSeries)

        returnSeries = str2double( ...
            string(returnSeries));

    end

    returnSeries = returnSeries(:);

    trainReturns = ...
        returnSeries(trainIdx);

    trainReturns = ...
        trainReturns(isfinite(trainReturns));

    [arIntercept,arCoefficients] = ...
        fitARModel( ...
            trainReturns, ...
            lagOrder);

    % Validation
    validationHistory = ...
        returnSeries(1:valIdx(1)-1);

    predVal = recursiveARPrediction( ...
        validationHistory, ...
        numel(valIdx), ...
        lagOrder, ...
        arIntercept, ...
        arCoefficients);

    % Test
    testHistory = ...
        returnSeries(1:testIdx(1)-1);

    predTest = recursiveARPrediction( ...
        testHistory, ...
        numel(testIdx), ...
        lagOrder, ...
        arIntercept, ...
        arCoefficients);

    trainedModels{6} = struct( ...
        'Intercept',arIntercept, ...
        'Coefficients',arCoefficients, ...
        'LagOrder',lagOrder);

    valPredictions{6} = predVal;

    testPredictions{6} = predTest;

    [valRMSE(6),valMAE(6),valR2(6)] = ...
        calculateMetrics(yVal,predVal);

    [testRMSE(6),testMAE(6),testR2(6)] = ...
        calculateMetrics(yTest,predTest);

    fprintf('Lag order: %d\n',lagOrder);

    printMetrics( ...
        valRMSE(6), ...
        valMAE(6), ...
        valR2(6), ...
        testRMSE(6), ...
        testMAE(6), ...
        testR2(6));

catch ME

    warning('step4:arFailed', ...
        'AR model failed: %s',ME.message);

end

%% ============================================================
% 16. RESULTS TABLE
% ============================================================

resultsTable = table( ...
    string(modelNames(:)), ...
    valRMSE, ...
    valMAE, ...
    valR2, ...
    testRMSE, ...
    testMAE, ...
    testR2, ...
    'VariableNames', ...
    { ...
    'Model', ...
    'ValidationRMSE', ...
    'ValidationMAE', ...
    'ValidationR2', ...
    'TestRMSE', ...
    'TestMAE', ...
    'TestR2'});

%% ============================================================
% 17. DISPLAY RESULTS
% ============================================================

fprintf('\n============================================================\n');
fprintf(' [MODEL COMPARISON RESULTS]\n');
fprintf('============================================================\n\n');

disp(resultsTable);

%% ============================================================
% 18. SELECT BEST MODEL
% ============================================================
%
% IMPORTANT:
% Test data is NOT used for model selection.

validModels = ...
    isfinite(valRMSE);

if any(validModels)

    temp = valRMSE;

    temp(~validModels) = Inf;

    [~,bestIdx] = min(temp);

    bestModelName = modelNames{bestIdx};

else

    bestIdx = NaN;

    bestModelName = 'None';

end

fprintf('\n============================================================\n');
fprintf(' [MODEL SELECTION]\n');
fprintf('============================================================\n');

fprintf('Selection criterion: Validation RMSE\n');

fprintf('Best model: %s\n',bestModelName);

if isfinite(bestIdx)

    fprintf('Validation RMSE: %.6f\n', ...
        valRMSE(bestIdx));

    fprintf('Test RMSE: %.6f\n', ...
        testRMSE(bestIdx));

end

fprintf('============================================================\n\n');

%% ============================================================
% 19. SAVE RESULTS TABLE
% ============================================================

tableFile = fullfile( ...
    tableDir, ...
    'model_comparison.csv');

fprintf('-> Saving comparison table...\n');

try

    writetable( ...
        resultsTable, ...
        tableFile);

catch ME

    warning('step4:tableSaveFailed', ...
        'Could not save comparison table: %s',ME.message);

end

%% ============================================================
% 20. SAVE MODEL DATA
% ============================================================

modelFile = fullfile( ...
    modelDir, ...
    'model_comparison.mat');

fprintf('-> Saving model results...\n');

try

    save( ...
        modelFile, ...
        'resultsTable', ...
        'modelNames', ...
        'trainedModels', ...
        'candidateFeatures', ...
        'mu', ...
        'sigma', ...
        'featureMedian', ...
        'bestIdx', ...
        'bestModelName', ...
        'valPredictions', ...
        'testPredictions', ...
        'trainIdx', ...
        'valIdx', ...
        'testIdx');

catch ME

    warning('step4:modelSaveFailed', ...
        'Could not save model results: %s',ME.message);

end

%% ============================================================
% 21. MODEL COMPARISON FIGURE
% ============================================================

figureFile = fullfile( ...
    figureDir, ...
    'model_comparison.png');

fprintf('-> Generating model comparison figure...\n');

try

    validPlot = isfinite(testRMSE);

    if any(validPlot)

        fig = figure( ...
            'Visible','off', ...
            'Position',[100 100 1200 700]);

        plotValues = ...
            testRMSE(validPlot);

        plotNames = ...
            modelNames(validPlot);

        bar(plotValues);

        set(gca, ...
            'XTick',1:numel(plotNames), ...
            'XTickLabel',plotNames);

        ylabel('Test RMSE');

        title('Model Comparison - Test RMSE');

        grid on;

        exportgraphics( ...
            fig, ...
            figureFile, ...
            'Resolution',150);

        close(fig);

    end

catch ME

    warning('step4:plotFailed', ...
        'Model comparison figure failed: %s',ME.message);

end

%% ============================================================
% 22. FINAL OUTPUT
% ============================================================

fprintf('\n============================================================\n');
fprintf(' [STEP 4 COMPLETE]\n');
fprintf('============================================================\n');

fprintf('Best model: %s\n',bestModelName);

fprintf('\nComparison table:\n%s\n', ...
    tableFile);

fprintf('\nModel results:\n%s\n', ...
    modelFile);

fprintf('\nComparison figure:\n%s\n', ...
    figureFile);

fprintf('\nProceeding to Step 5...\n\n');

end


%% ============================================================
function [rmse,mae,r2] = calculateMetrics(yTrue,yPred)
%CALCULATEMETRICS Calculate RMSE, MAE and R2.

yTrue = yTrue(:);

yPred = yPred(:);

valid = ...
    isfinite(yTrue) & ...
    isfinite(yPred);

yTrue = yTrue(valid);

yPred = yPred(valid);

if isempty(yTrue)

    rmse = NaN;
    mae = NaN;
    r2 = NaN;

    return;

end

errors = ...
    yTrue - yPred;

rmse = sqrt( ...
    mean(errors.^2));

mae = mean( ...
    abs(errors));

ssResidual = ...
    sum(errors.^2);

ssTotal = ...
    sum((yTrue - mean(yTrue)).^2);

if ssTotal > 0

    r2 = ...
        1 - ssResidual / ssTotal;

else

    r2 = NaN;

end

end


%% ============================================================
function printMetrics(valRMSE,valMAE,valR2, ...
                       testRMSE,testMAE,testR2)

fprintf('Validation RMSE: %.6f\n',valRMSE);

fprintf('Validation MAE : %.6f\n',valMAE);

fprintf('Validation R2  : %.4f\n',valR2);

fprintf('Test RMSE      : %.6f\n',testRMSE);

fprintf('Test MAE       : %.6f\n',testMAE);

fprintf('Test R2        : %.4f\n',testR2);

end


%% ============================================================
function pred = extractSequencePrediction(output,n)
%EXTRACTSEQUENCEPREDICTION Convert sequence output to vector.

if iscell(output)

    output = output{1};

end

pred = output(:);

if numel(pred) >= n

    pred = pred(end-n+1:end);

else

    result = nan(n,1);

    result(end-numel(pred)+1:end) = pred;

    pred = result;

end

end


%% ============================================================
function [intercept,coefficients] = fitARModel(y,p)
%FITARMODEL Fit AR(p) using ordinary least squares.

y = y(:);

y = y(isfinite(y));

n = numel(y);

if n <= p + 10

    error( ...
        'Not enough observations for AR model.');

end

rows = n-p;

X = ones(rows,p+1);

Y = y(p+1:end);

for lag = 1:p

    X(:,lag+1) = ...
        y(p+1-lag:end-lag);

end

valid = ...
    isfinite(Y) & ...
    all(isfinite(X),2);

X = X(valid,:);

Y = Y(valid);

if size(X,1) <= p

    error( ...
        'Not enough valid rows for AR fitting.');

end

beta = pinv(X) * Y;

intercept = beta(1);

coefficients = beta(2:end);

end


%% ============================================================
function predictions = recursiveARPrediction( ...
    history, ...
    horizon, ...
    p, ...
    intercept, ...
    coefficients)
%RECURSIVEARPREDICTION Generate recursive AR predictions.

history = history(:);

history = history(isfinite(history));

if numel(history) < p

    error( ...
        'Insufficient history for AR prediction.');

end

predictions = nan(horizon,1);

for k = 1:horizon

    recent = ...
        history(end-p+1:end);

    recent = flipud(recent);

    predictions(k) = ...
        intercept + ...
        coefficients(:)' * recent;

    history(end+1,1) = ...
        predictions(k); %#ok<AGROW>

end

end