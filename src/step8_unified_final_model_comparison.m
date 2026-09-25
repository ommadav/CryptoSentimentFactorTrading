%% STEP 8 - UNIFIED FINAL MODEL COMPARISON
% Crypto Sentiment Factor Trading
%
% Models:
%   1. AR(5)
%   2. Ridge Regression
%   3. Bagged Regression Trees
%   4. Random Forest Regression
%   5. LSBoost Regression
%
% IMPORTANT:
% - Same target for all models
% - Same fixed train/validation/test dates
% - Validation used for model/threshold selection
% - Test is completely untouched until final evaluation
% - Same transaction cost
% - Same Sharpe annualization
% - Same trading logic
%
% Output:
%   results/tables/STEP8_UNIFIED_MODEL_COMPARISON.csv
%   results/tables/STEP8_UNIFIED_MODEL_COMPARISON_SUMMARY.csv
%   results/tables/STEP8_MODEL_THRESHOLDS.csv
%   results/tables/STEP8_MODEL_PREDICTIONS.csv
%   results/tables/STEP8_FACTOR_LIST.csv
%   results/models/STEP8_UNIFIED_MODELS.mat
%   results/figures/STEP8_MODEL_RETURN_COMPARISON.png
%   results/figures/STEP8_MODEL_EQUITY_CURVES.png
%
% Research/backtesting only. Not live trading advice.

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' STEP 8 - UNIFIED FINAL MODEL COMPARISON\n');
fprintf('============================================================\n');

%% ------------------------------------------------------------
% 1. PATH SETUP
% ------------------------------------------------------------

thisFile = mfilename('fullpath');
if isempty(thisFile)
    projectRoot = pwd;
else
    srcFolder = fileparts(thisFile);
    projectRoot = fileparts(srcFolder);
end

dataFile = fullfile(projectRoot, 'data', 'processed', 'factor_dataset.csv');

% Fallback location
if ~isfile(dataFile)
    dataFile = fullfile(projectRoot, 'data', 'raw', 'factor_dataset.csv');
end

if ~isfile(dataFile)
    error('factor_dataset.csv not found.');
end

resultsTableDir = fullfile(projectRoot, 'results', 'tables');
resultsModelDir = fullfile(projectRoot, 'results', 'models');
resultsFigureDir = fullfile(projectRoot, 'results', 'figures');

if ~exist(resultsTableDir, 'dir')
    mkdir(resultsTableDir);
end

if ~exist(resultsModelDir, 'dir')
    mkdir(resultsModelDir);
end

if ~exist(resultsFigureDir, 'dir')
    mkdir(resultsFigureDir);
end

fprintf('\nProject root:\n%s\n', projectRoot);
fprintf('Data file:\n%s\n', dataFile);

%% ------------------------------------------------------------
% 2. EXPERIMENT SETTINGS
% ------------------------------------------------------------

% Fixed date boundaries.
% These are intentionally kept identical across models.

trainStart = datetime(2018,1,6);
trainEnd   = datetime(2020,4,10);

valStart   = datetime(2020,4,11);
valEnd     = datetime(2020,10,4);

testStart  = datetime(2020,10,5);
testEnd    = datetime(2021,3,31);

% Trading parameters
initialCapital = 10000;
feeRate = 0.001;                 % 0.10%
annualFactor = sqrt(365);

% Threshold search
thresholdGrid = (0.0001:0.0001:0.0100)';
minimumValidationTrades = 4;

% Ridge lambda grid
ridgeLambdaGrid = logspace(-6,2,17);

% Randomness control
rng(42, 'twister');

fprintf('\n------------------------------------------------------------\n');
fprintf('FIXED EXPERIMENT PERIODS\n');
fprintf('------------------------------------------------------------\n');

fprintf('Training   : %s -> %s\n', ...
    datestr(trainStart,'yyyy-mm-dd'), ...
    datestr(trainEnd,'yyyy-mm-dd'));

fprintf('Validation : %s -> %s\n', ...
    datestr(valStart,'yyyy-mm-dd'), ...
    datestr(valEnd,'yyyy-mm-dd'));

fprintf('Test       : %s -> %s\n', ...
    datestr(testStart,'yyyy-mm-dd'), ...
    datestr(testEnd,'yyyy-mm-dd'));

fprintf('Transaction cost : %.2f%%\n', feeRate*100);
fprintf('Minimum validation trades : %d\n', minimumValidationTrades);

%% ------------------------------------------------------------
% 3. LOAD DATA
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('LOADING DATA\n');
fprintf('============================================================\n');

T = readtable(dataFile, 'VariableNamingRule','preserve');

fprintf('Rows    : %d\n', height(T));
fprintf('Columns : %d\n', width(T));

%% ------------------------------------------------------------
% 4. IDENTIFY DATE COLUMN
% ------------------------------------------------------------

dateVarCandidates = {'Date','date','Open time','time','Timestamp'};

dateVar = '';

for i = 1:numel(dateVarCandidates)

    idx = find(strcmpi(string(T.Properties.VariableNames), ...
        dateVarCandidates{i}), 1);

    if ~isempty(idx)
        dateVar = T.Properties.VariableNames{idx};
        break;
    end

end

if isempty(dateVar)
    error('Could not identify Date column.');
end

rawDate = T.(dateVar);

if isdatetime(rawDate)

    dates = rawDate;

elseif isnumeric(rawDate)

    % MATLAB Excel-style serial date
    try
        dates = datetime(rawDate, 'ConvertFrom','excel');
    catch
        dates = datetime(rawDate, 'ConvertFrom','posixtime');
    end

else

    rawDateString = string(rawDate);

    formats = { ...
        'yyyy-MM-dd', ...
        'yyyy-MM-dd HH:mm:ss', ...
        'yyyy-MM-dd HH:mm:ss.SSS', ...
        'dd-MM-yyyy', ...
        'MM/dd/yyyy', ...
        'yyyy/MM/dd'};

    dates = NaT(size(rawDateString));

    for f = 1:numel(formats)

        missing = isnat(dates);

        if ~any(missing)
            break;
        end

        try
            temp = datetime(rawDateString(missing), ...
                'InputFormat',formats{f});

            dates(missing) = temp;
        catch
        end

    end

    % Final generic parser
    missing = isnat(dates);

    if any(missing)
        try
            dates(missing) = datetime(rawDateString(missing));
        catch
        end
    end

end

T.Date = dates;

%% ------------------------------------------------------------
% 5. SORT AND REMOVE DUPLICATES
% ------------------------------------------------------------

validDate = ~isnat(T.Date);

T = T(validDate,:);

T = sortrows(T,'Date');

[~, uniqueIdx] = unique(T.Date,'stable');

T = T(uniqueIdx,:);

fprintf('\nAfter date cleaning:\n');
fprintf('Rows: %d\n', height(T));
fprintf('Date range: %s -> %s\n', ...
    datestr(min(T.Date),'yyyy-mm-dd'), ...
    datestr(max(T.Date),'yyyy-mm-dd'));

%% ------------------------------------------------------------
% 6. IDENTIFY TARGET AND CLOSE
% ------------------------------------------------------------

targetName = 'TargetReturn1D';

if ~ismember(targetName, T.Properties.VariableNames)
    error('TargetReturn1D not found.');
end

closeCandidates = {'Close','close','BTC_Close','Price'};

closeName = '';

for i = 1:numel(closeCandidates)

    idx = find(strcmpi(string(T.Properties.VariableNames), ...
        closeCandidates{i}),1);

    if ~isempty(idx)
        closeName = T.Properties.VariableNames{idx};
        break;
    end

end

if isempty(closeName)
    error('Close price column not found.');
end

target = double(T.(targetName));
closePrice = double(T.(closeName));

%% ------------------------------------------------------------
% 7. LOAD EXACT STEP-7 FACTOR LIST IF AVAILABLE
% ------------------------------------------------------------

factorListFile = fullfile( ...
    resultsTableDir, ...
    'STEP7_RIDGE_FACTOR_LIST.csv');

factorNames = {};

if isfile(factorListFile)

    fprintf('\nUsing factor list generated by Step 7:\n%s\n', ...
        factorListFile);

    F = readtable(factorListFile, ...
        'VariableNamingRule','preserve');

    if width(F) >= 1

        factorNames = cellstr(string(F{:,1}));

    end

end

%% ------------------------------------------------------------
% 8. FALLBACK FACTOR DETECTION
% ------------------------------------------------------------

if isempty(factorNames)

    fprintf('\nStep 7 factor list not found.\n');
    fprintf('Automatically detecting factors...\n');

    allNames = T.Properties.VariableNames;

    excluded = { ...
        'Date', ...
        'TargetReturn1D', ...
        'TargetDirection', ...
        'Open', ...
        'High', ...
        'Low', ...
        'Close', ...
        'Volume', ...
        'Open time', ...
        'Close time', ...
        'Quote asset volume', ...
        'Number of trades', ...
        'Taker buy base asset volume', ...
        'Taker buy quote asset volume', ...
        'Ignore', ...
        'SampleSize'};

    for i = 1:numel(allNames)

        name = allNames{i};

        if any(strcmpi(name,excluded))
            continue;
        end

        x = T.(name);

        if isnumeric(x) || islogical(x)

            if numel(unique(x(~isnan(double(x))))) > 1
                factorNames{end+1} = name; %#ok<SAGROW>
            end

        end

    end

end

%% ------------------------------------------------------------
% 9. REMOVE INVALID FACTOR NAMES
% ------------------------------------------------------------

factorNames = factorNames( ...
    ismember(factorNames,T.Properties.VariableNames));

% Remove duplicate factor names
factorNames = unique(factorNames,'stable');

fprintf('\n============================================================\n');
fprintf('FACTOR SET\n');
fprintf('============================================================\n');

fprintf('Number of common factors: %d\n',numel(factorNames));

for i = 1:numel(factorNames)
    fprintf('%3d. %s\n',i,factorNames{i});
end

%% Save factor list

factorTable = table(string(factorNames(:)), ...
    'VariableNames',{'Factor'});

writetable(factorTable, ...
    fullfile(resultsTableDir,'STEP8_FACTOR_LIST.csv'));

%% ------------------------------------------------------------
% 10. BUILD COMMON FACTOR MATRIX
% ------------------------------------------------------------

X = NaN(height(T),numel(factorNames));

for j = 1:numel(factorNames)

    x = T.(factorNames{j});

    if iscell(x)
        x = str2double(string(x));
    elseif isstring(x) || ischar(x) || iscategorical(x)
        x = str2double(string(x));
    end

    X(:,j) = double(x);

end

%% ------------------------------------------------------------
% 11. VALIDATE TARGET
% ------------------------------------------------------------

target = double(target);

% Remove invalid infinite values
target(~isfinite(target)) = NaN;
closePrice(~isfinite(closePrice)) = NaN;

%% ------------------------------------------------------------
% 12. COMMON DATE WINDOWS
% ------------------------------------------------------------

idxTrainDate = T.Date >= trainStart & T.Date <= trainEnd;
idxValDate   = T.Date >= valStart   & T.Date <= valEnd;
idxTestDate  = T.Date >= testStart  & T.Date <= testEnd;

fprintf('\nRows before feature cleaning:\n');
fprintf('Train : %d\n',sum(idxTrainDate));
fprintf('Val   : %d\n',sum(idxValDate));
fprintf('Test  : %d\n',sum(idxTestDate));

%% ------------------------------------------------------------
% 13. COMMON FEATURE CLEANING
% ------------------------------------------------------------

validFeatures = all(isfinite(X),2) & isfinite(target);

fprintf('\nRows removed because of missing/invalid factor values: %d\n', ...
    sum(~validFeatures));

% For factor models
idxTrainFactor = idxTrainDate & validFeatures;
idxValFactor   = idxValDate   & validFeatures;
idxTestFactor  = idxTestDate  & validFeatures;

fprintf('\nFinal factor-model observations:\n');
fprintf('Train : %d\n',sum(idxTrainFactor));
fprintf('Val   : %d\n',sum(idxValFactor));
fprintf('Test  : %d\n',sum(idxTestFactor));

%% ------------------------------------------------------------
% 14. AR(5) DATASET
% ------------------------------------------------------------

% AR(5) uses only historical returns.
%
% Return(t) is constructed from Close(t)/Close(t-1)-1.
%
% Target(t) = Close(t+1)/Close(t)-1.
%
% The five lag variables therefore contain information available
% before the target period.

actualReturn = [NaN; closePrice(2:end)./closePrice(1:end-1)-1];

AR5 = NaN(height(T),5);

for lag = 1:5
    AR5(:,lag) = lagmatrix(actualReturn,lag);
end

% If lagmatrix is unavailable, manually construct it.
if all(isnan(AR5(:)))

    for lag = 1:5
        AR5((lag+1):end,lag) = actualReturn(1:end-lag);
    end

end

validAR = all(isfinite(AR5),2) & isfinite(target);

idxTrainAR = idxTrainDate & validAR;
idxValAR   = idxValDate   & validAR;
idxTestAR  = idxTestDate  & validAR;

fprintf('\nAR(5) observations:\n');
fprintf('Train : %d\n',sum(idxTrainAR));
fprintf('Val   : %d\n',sum(idxValAR));
fprintf('Test  : %d\n',sum(idxTestAR));

%% ------------------------------------------------------------
% 15. STORAGE
% ------------------------------------------------------------

modelNames = { ...
    'AR5'; ...
    'Ridge'; ...
    'BaggedRegression'; ...
    'RandomForest'; ...
    'LSBoost'};

nModels = numel(modelNames);

modelObjects = cell(nModels,1);

validationThreshold = NaN(nModels,1);
validationRMSE = NaN(nModels,1);
validationMAE = NaN(nModels,1);
validationR2 = NaN(nModels,1);
validationDirectional = NaN(nModels,1);
validationSharpe = NaN(nModels,1);
validationReturn = NaN(nModels,1);
validationMaxDD = NaN(nModels,1);
validationTrades = NaN(nModels,1);

testRMSE = NaN(nModels,1);
testMAE = NaN(nModels,1);
testMSE = NaN(nModels,1);
testR2 = NaN(nModels,1);
testDirectional = NaN(nModels,1);
testPredictionCorrelation = NaN(nModels,1);

testFinalCapital = NaN(nModels,1);
testReturn = NaN(nModels,1);
testSharpe = NaN(nModels,1);
testSortino = NaN(nModels,1);
testMaxDD = NaN(nModels,1);
testWinRate = NaN(nModels,1);
testProfitFactor = NaN(nModels,1);
testTrades = NaN(nModels,1);

testBuy = NaN(nModels,1);
testHold = NaN(nModels,1);
testSell = NaN(nModels,1);

%% ------------------------------------------------------------
% 16. PREDICTION STORAGE
% ------------------------------------------------------------

predictionRows = [];

%% ------------------------------------------------------------
% 17. FUNCTION FOR REGRESSION METRICS
% ------------------------------------------------------------

calcRMSE = @(a,p) sqrt(mean((a-p).^2,'omitnan'));
calcMAE  = @(a,p) mean(abs(a-p),'omitnan');

%% ------------------------------------------------------------
% 18. MODEL 1 - AR(5)
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('MODEL 1/5 - AR(5)\n');
fprintf('============================================================\n');

Xtr = AR5(idxTrainAR,:);
Ytr = target(idxTrainAR);

Xva = AR5(idxValAR,:);
Yva = target(idxValAR);

Xte = AR5(idxTestAR,:);
Yte = target(idxTestAR);

% Standardization using training data only
muAR = mean(Xtr,1,'omitnan');
sigmaAR = std(Xtr,0,1,'omitnan');

sigmaAR(sigmaAR == 0 | ~isfinite(sigmaAR)) = 1;

XtrS = (Xtr-muAR)./sigmaAR;
XvaS = (Xva-muAR)./sigmaAR;
XteS = (Xte-muAR)./sigmaAR;

ARModel = fitrlinear( ...
    XtrS, ...
    Ytr, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda',1e-4);

predValAR = predict(ARModel,XvaS);
predTestAR = predict(ARModel,XteS);

% Refit on train + validation
Xdev = [XtrS; XvaS];
Ydev = [Ytr; Yva];

ARFinal = fitrlinear( ...
    Xdev, ...
    Ydev, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda',1e-4);

predTestAR = predict(ARFinal,XteS);

modelObjects{1} = struct( ...
    'model',ARFinal, ...
    'mu',muAR, ...
    'sigma',sigmaAR);

%% ------------------------------------------------------------
% 19. AR VALIDATION METRICS
% ------------------------------------------------------------

validationRMSE(1) = calcRMSE(Yva,predValAR);
validationMAE(1) = calcMAE(Yva,predValAR);

sst = sum((Yva-mean(Yva)).^2);
sse = sum((Yva-predValAR).^2);

validationR2(1) = 1-sse/max(sst,eps);

validationDirectional(1) = ...
    mean(sign(Yva)==sign(predValAR))*100;

%% ------------------------------------------------------------
% 20. AR THRESHOLD SELECTION
% ------------------------------------------------------------

bestFound = false;
bestSharpe = -Inf;
bestReturn = -Inf;
bestThreshold = thresholdGrid(1);

for k = 1:numel(thresholdGrid)

    th = thresholdGrid(k);

    pos = zeros(size(predValAR));

    pos(predValAR > th) = 1;
    pos(predValAR < -th) = -1;

    [ret,eq,stats] = runBacktest( ...
        Yva,pos,initialCapital,feeRate,annualFactor);

    if stats.trades < minimumValidationTrades
        continue;
    end

    if stats.sharpe > bestSharpe || ...
            (abs(stats.sharpe-bestSharpe)<1e-12 && ret>bestReturn)

        bestSharpe = stats.sharpe;
        bestReturn = ret;
        bestThreshold = th;
        bestFound = true;

    end

end

if ~bestFound

    fprintf('WARNING: No AR threshold met minimum trade requirement.\n');
    bestThreshold = 0.0001;
end

validationThreshold(1) = bestThreshold;

posValAR = makePosition(predValAR,bestThreshold);

[valRetAR,~,valStatsAR] = runBacktest( ...
    Yva,posValAR,initialCapital,feeRate,annualFactor);

validationSharpe(1) = valStatsAR.sharpe;
validationReturn(1) = valRetAR;
validationMaxDD(1) = valStatsAR.maxDD;
validationTrades(1) = valStatsAR.trades;

fprintf('Validation RMSE      : %.6f\n',validationRMSE(1));
fprintf('Validation MAE       : %.6f\n',validationMAE(1));
fprintf('Validation R2        : %.6f\n',validationR2(1));
fprintf('Directional accuracy : %.2f%%\n',validationDirectional(1));
fprintf('Selected threshold   : %.2f%%\n',bestThreshold*100);
fprintf('Validation return    : %.2f%%\n',validationReturn(1)*100);
fprintf('Validation Sharpe    : %.4f\n',validationSharpe(1));
fprintf('Validation trades    : %d\n',validationTrades(1));

%% ------------------------------------------------------------
% 21. MODEL 2 - RIDGE
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('MODEL 2/5 - RIDGE REGRESSION\n');
fprintf('============================================================\n');

Xtr = X(idxTrainFactor,:);
Ytr = target(idxTrainFactor);

Xva = X(idxValFactor,:);
Yva = target(idxValFactor);

Xte = X(idxTestFactor,:);
Yte = target(idxTestFactor);

% Training-only standardization
mu = mean(Xtr,1,'omitnan');
sigma = std(Xtr,0,1,'omitnan');

sigma(sigma == 0 | ~isfinite(sigma)) = 1;

XtrS = (Xtr-mu)./sigma;
XvaS = (Xva-mu)./sigma;
XteS = (Xte-mu)./sigma;

bestLambda = NaN;
bestRMSE = Inf;
bestRidgeModel = [];

for l = 1:numel(ridgeLambdaGrid)

    lambda = ridgeLambdaGrid(l);

    mdl = fitrlinear( ...
        XtrS,Ytr, ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda',lambda);

    p = predict(mdl,XvaS);

    rmse = calcRMSE(Yva,p);

    if rmse < bestRMSE
        bestRMSE = rmse;
        bestLambda = lambda;
        bestRidgeModel = mdl;
    end

end

fprintf('Selected Ridge lambda: %.8f\n',bestLambda);

predValRidge = predict(bestRidgeModel,XvaS);

% Refit on train + validation
Xdev = [XtrS;XvaS];
Ydev = [Ytr;Yva];

RidgeFinal = fitrlinear( ...
    Xdev,Ydev, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda',bestLambda);

predTestRidge = predict(RidgeFinal,XteS);

modelObjects{2} = struct( ...
    'model',RidgeFinal, ...
    'mu',mu, ...
    'sigma',sigma, ...
    'lambda',bestLambda);

%% Ridge metrics

validationRMSE(2) = calcRMSE(Yva,predValRidge);
validationMAE(2) = calcMAE(Yva,predValRidge);

sst = sum((Yva-mean(Yva)).^2);
sse = sum((Yva-predValRidge).^2);

validationR2(2) = 1-sse/max(sst,eps);

validationDirectional(2) = ...
    mean(sign(Yva)==sign(predValRidge))*100;

%% Ridge threshold

bestFound = false;
bestSharpe = -Inf;
bestReturn = -Inf;
bestThreshold = thresholdGrid(1);

for k = 1:numel(thresholdGrid)

    th = thresholdGrid(k);

    pos = makePosition(predValRidge,th);

    [ret,~,stats] = runBacktest( ...
        Yva,pos,initialCapital,feeRate,annualFactor);

    if stats.trades < minimumValidationTrades
        continue;
    end

    if stats.sharpe > bestSharpe || ...
            (abs(stats.sharpe-bestSharpe)<1e-12 && ret>bestReturn)

        bestSharpe = stats.sharpe;
        bestReturn = ret;
        bestThreshold = th;
        bestFound = true;

    end

end

if ~bestFound
    bestThreshold = 0.0001;
end

validationThreshold(2) = bestThreshold;

posValRidge = makePosition(predValRidge,bestThreshold);

[valRet,~,valStats] = runBacktest( ...
    Yva,posValRidge,initialCapital,feeRate,annualFactor);

validationSharpe(2) = valStats.sharpe;
validationReturn(2) = valRet;
validationMaxDD(2) = valStats.maxDD;
validationTrades(2) = valStats.trades;

fprintf('Validation RMSE      : %.6f\n',validationRMSE(2));
fprintf('Validation MAE       : %.6f\n',validationMAE(2));
fprintf('Validation R2        : %.6f\n',validationR2(2));
fprintf('Directional accuracy : %.2f%%\n',validationDirectional(2));
fprintf('Selected threshold   : %.2f%%\n',bestThreshold*100);
fprintf('Validation return    : %.2f%%\n',validationReturn(2)*100);
fprintf('Validation Sharpe    : %.4f\n',validationSharpe(2));
fprintf('Validation trades    : %d\n',validationTrades(2));

%% ------------------------------------------------------------
% 22. MODEL 3 - BAGGED REGRESSION
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('MODEL 3/5 - BAGGED REGRESSION TREES\n');
fprintf('============================================================\n');

% Use raw factors for trees.
% Trees do not require standardization.

Xtr = X(idxTrainFactor,:);
Ytr = target(idxTrainFactor);

Xva = X(idxValFactor,:);
Yva = target(idxValFactor);

Xte = X(idxTestFactor,:);
Yte = target(idxTestFactor);

treeTemplate = templateTree( ...
    'MinLeafSize',10);

BagModel = fitrensemble( ...
    Xtr,Ytr, ...
    'Method','Bag', ...
    'NumLearningCycles',100, ...
    'Learners',treeTemplate);

predValBag = predict(BagModel,Xva);

%% Validation metrics

validationRMSE(3) = calcRMSE(Yva,predValBag);
validationMAE(3) = calcMAE(Yva,predValBag);

sst = sum((Yva-mean(Yva)).^2);
sse = sum((Yva-predValBag).^2);

validationR2(3) = 1-sse/max(sst,eps);

validationDirectional(3) = ...
    mean(sign(Yva)==sign(predValBag))*100;

%% Threshold

bestFound = false;
bestSharpe = -Inf;
bestReturn = -Inf;
bestThreshold = thresholdGrid(1);

for k = 1:numel(thresholdGrid)

    th = thresholdGrid(k);

    pos = makePosition(predValBag,th);

    [ret,~,stats] = runBacktest( ...
        Yva,pos,initialCapital,feeRate,annualFactor);

    if stats.trades < minimumValidationTrades
        continue;
    end

    if stats.sharpe > bestSharpe || ...
            (abs(stats.sharpe-bestSharpe)<1e-12 && ret>bestReturn)

        bestSharpe = stats.sharpe;
        bestReturn = ret;
        bestThreshold = th;
        bestFound = true;

    end

end

if ~bestFound
    bestThreshold = 0.0001;
end

validationThreshold(3) = bestThreshold;

posValBag = makePosition(predValBag,bestThreshold);

[valRet,~,valStats] = runBacktest( ...
    Yva,posValBag,initialCapital,feeRate,annualFactor);

validationSharpe(3) = valStats.sharpe;
validationReturn(3) = valRet;
validationMaxDD(3) = valStats.maxDD;
validationTrades(3) = valStats.trades;

%% Refit train + validation

Xdev = [Xtr;Xva];
Ydev = [Ytr;Yva];

BagFinal = fitrensemble( ...
    Xdev,Ydev, ...
    'Method','Bag', ...
    'NumLearningCycles',100, ...
    'Learners',treeTemplate);

predTestBag = predict(BagFinal,Xte);

modelObjects{3} = BagFinal;

fprintf('Validation RMSE      : %.6f\n',validationRMSE(3));
fprintf('Validation MAE       : %.6f\n',validationMAE(3));
fprintf('Validation R2        : %.6f\n',validationR2(3));
fprintf('Directional accuracy : %.2f%%\n',validationDirectional(3));
fprintf('Selected threshold   : %.2f%%\n',bestThreshold*100);
fprintf('Validation return    : %.2f%%\n',validationReturn(3)*100);
fprintf('Validation Sharpe    : %.4f\n',validationSharpe(3));
fprintf('Validation trades    : %d\n',validationTrades(3));

%% ------------------------------------------------------------
% 23. MODEL 4 - RANDOM FOREST REGRESSION
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('MODEL 4/5 - RANDOM FOREST REGRESSION\n');
fprintf('============================================================\n');

% TreeBagger provides a conventional Random Forest-style
% regression implementation.

numTrees = 200;

numPredictors = size(Xtr,2);

numSample = max(1,round(sqrt(numPredictors)));

RFModel = TreeBagger( ...
    numTrees, ...
    Xtr,Ytr, ...
    'Method','regression', ...
    'MinLeafSize',5, ...
    'NumPredictorsToSample',numSample, ...
    'OOBPrediction','off');

predValRF = predict(RFModel,Xva);

predValRF = double(predValRF);

%% Validation metrics

validationRMSE(4) = calcRMSE(Yva,predValRF);
validationMAE(4) = calcMAE(Yva,predValRF);

sst = sum((Yva-mean(Yva)).^2);
sse = sum((Yva-predValRF).^2);

validationR2(4) = 1-sse/max(sst,eps);

validationDirectional(4) = ...
    mean(sign(Yva)==sign(predValRF))*100;

%% Threshold

bestFound = false;
bestSharpe = -Inf;
bestReturn = -Inf;
bestThreshold = thresholdGrid(1);

for k = 1:numel(thresholdGrid)

    th = thresholdGrid(k);

    pos = makePosition(predValRF,th);

    [ret,~,stats] = runBacktest( ...
        Yva,pos,initialCapital,feeRate,annualFactor);

    if stats.trades < minimumValidationTrades
        continue;
    end

    if stats.sharpe > bestSharpe || ...
            (abs(stats.sharpe-bestSharpe)<1e-12 && ret>bestReturn)

        bestSharpe = stats.sharpe;
        bestReturn = ret;
        bestThreshold = th;
        bestFound = true;

    end

end

if ~bestFound
    bestThreshold = 0.0001;
end

validationThreshold(4) = bestThreshold;

posValRF = makePosition(predValRF,bestThreshold);

[valRet,~,valStats] = runBacktest( ...
    Yva,posValRF,initialCapital,feeRate,annualFactor);

validationSharpe(4) = valStats.sharpe;
validationReturn(4) = valRet;
validationMaxDD(4) = valStats.maxDD;
validationTrades(4) = valStats.trades;

%% Refit on train + validation

Xdev = [Xtr;Xva];
Ydev = [Ytr;Yva];

RFFinal = TreeBagger( ...
    numTrees, ...
    Xdev,Ydev, ...
    'Method','regression', ...
    'MinLeafSize',5, ...
    'NumPredictorsToSample',numSample, ...
    'OOBPrediction','off');

predTestRF = predict(RFFinal,Xte);

predTestRF = double(predTestRF);

modelObjects{4} = RFFinal;

fprintf('Validation RMSE      : %.6f\n',validationRMSE(4));
fprintf('Validation MAE       : %.6f\n',validationMAE(4));
fprintf('Validation R2        : %.6f\n',validationR2(4));
fprintf('Directional accuracy : %.2f%%\n',validationDirectional(4));
fprintf('Selected threshold   : %.2f%%\n',bestThreshold*100);
fprintf('Validation return    : %.2f%%\n',validationReturn(4)*100);
fprintf('Validation Sharpe    : %.4f\n',validationSharpe(4));
fprintf('Validation trades    : %d\n',validationTrades(4));

%% ------------------------------------------------------------
% 24. MODEL 5 - LSBOOST
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('MODEL 5/5 - LSBOOST REGRESSION\n');
fprintf('============================================================\n');

boostTemplate = templateTree( ...
    'MinLeafSize',10);

LSModel = fitrensemble( ...
    Xtr,Ytr, ...
    'Method','LSBoost', ...
    'NumLearningCycles',100, ...
    'LearnRate',0.05, ...
    'Learners',boostTemplate);

predValLS = predict(LSModel,Xva);

%% Validation metrics

validationRMSE(5) = calcRMSE(Yva,predValLS);
validationMAE(5) = calcMAE(Yva,predValLS);

sst = sum((Yva-mean(Yva)).^2);
sse = sum((Yva-predValLS).^2);

validationR2(5) = 1-sse/max(sst,eps);

validationDirectional(5) = ...
    mean(sign(Yva)==sign(predValLS))*100;

%% Threshold

bestFound = false;
bestSharpe = -Inf;
bestReturn = -Inf;
bestThreshold = thresholdGrid(1);

for k = 1:numel(thresholdGrid)

    th = thresholdGrid(k);

    pos = makePosition(predValLS,th);

    [ret,~,stats] = runBacktest( ...
        Yva,pos,initialCapital,feeRate,annualFactor);

    if stats.trades < minimumValidationTrades
        continue;
    end

    if stats.sharpe > bestSharpe || ...
            (abs(stats.sharpe-bestSharpe)<1e-12 && ret>bestReturn)

        bestSharpe = stats.sharpe;
        bestReturn = ret;
        bestThreshold = th;
        bestFound = true;

    end

end

if ~bestFound
    bestThreshold = 0.0001;
end

validationThreshold(5) = bestThreshold;

posValLS = makePosition(predValLS,bestThreshold);

[valRet,~,valStats] = runBacktest( ...
    Yva,posValLS,initialCapital,feeRate,annualFactor);

validationSharpe(5) = valStats.sharpe;
validationReturn(5) = valRet;
validationMaxDD(5) = valStats.maxDD;
validationTrades(5) = valStats.trades;

%% Refit

Xdev = [Xtr;Xva];
Ydev = [Ytr;Yva];

LSFinal = fitrensemble( ...
    Xdev,Ydev, ...
    'Method','LSBoost', ...
    'NumLearningCycles',100, ...
    'LearnRate',0.05, ...
    'Learners',boostTemplate);

predTestLS = predict(LSFinal,Xte);

modelObjects{5} = LSFinal;

fprintf('Validation RMSE      : %.6f\n',validationRMSE(5));
fprintf('Validation MAE       : %.6f\n',validationMAE(5));
fprintf('Validation R2        : %.6f\n',validationR2(5));
fprintf('Directional accuracy : %.2f%%\n',validationDirectional(5));
fprintf('Selected threshold   : %.2f%%\n',bestThreshold*100);
fprintf('Validation return    : %.2f%%\n',validationReturn(5)*100);
fprintf('Validation Sharpe    : %.4f\n',validationSharpe(5));
fprintf('Validation trades    : %d\n',validationTrades(5));

%% ------------------------------------------------------------
% 25. FINAL TEST EVALUATION
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('FINAL UNTOUCHED TEST EVALUATION\n');
fprintf('============================================================\n');

testPredictions = cell(nModels,1);
testActuals = cell(nModels,1);
testDates = cell(nModels,1);
testPositions = cell(nModels,1);
testEquity = cell(nModels,1);

%% ---------------- AR TEST ----------------

pred = predTestAR;
actual = Yte;
d = T.Date(idxTestAR);

threshold = validationThreshold(1);
position = makePosition(pred,threshold);

[ret,equity,stats] = runBacktest( ...
    actual,position,initialCapital,feeRate,annualFactor);

testPredictions{1} = pred;
testActuals{1} = actual;
testDates{1} = d;
testPositions{1} = position;
testEquity{1} = equity;

testRMSE(1) = calcRMSE(actual,pred);
testMAE(1) = calcMAE(actual,pred);
testMSE(1) = mean((actual-pred).^2,'omitnan');

sst = sum((actual-mean(actual)).^2);
sse = sum((actual-pred).^2);

testR2(1) = 1-sse/max(sst,eps);

testDirectional(1) = ...
    mean(sign(actual)==sign(pred))*100;

testPredictionCorrelation(1) = safeCorrelation(actual,pred);

testFinalCapital(1) = stats.finalCapital;
testReturn(1) = ret;
testSharpe(1) = stats.sharpe;
testSortino(1) = stats.sortino;
testMaxDD(1) = stats.maxDD;
testWinRate(1) = stats.winRate;
testProfitFactor(1) = stats.profitFactor;
testTrades(1) = stats.trades;

testBuy(1) = sum(position==1);
testHold(1) = sum(position==0);
testSell(1) = sum(position==-1);

%% ---------------- RIDGE TEST ----------------

pred = predTestRidge;
actual = Yte;
d = T.Date(idxTestFactor);

threshold = validationThreshold(2);
position = makePosition(pred,threshold);

[ret,equity,stats] = runBacktest( ...
    actual,position,initialCapital,feeRate,annualFactor);

testPredictions{2} = pred;
testActuals{2} = actual;
testDates{2} = d;
testPositions{2} = position;
testEquity{2} = equity;

testRMSE(2) = calcRMSE(actual,pred);
testMAE(2) = calcMAE(actual,pred);
testMSE(2) = mean((actual-pred).^2,'omitnan');

sst = sum((actual-mean(actual)).^2);
sse = sum((actual-pred).^2);

testR2(2) = 1-sse/max(sst,eps);

testDirectional(2) = ...
    mean(sign(actual)==sign(pred))*100;

testPredictionCorrelation(2) = safeCorrelation(actual,pred);

testFinalCapital(2) = stats.finalCapital;
testReturn(2) = ret;
testSharpe(2) = stats.sharpe;
testSortino(2) = stats.sortino;
testMaxDD(2) = stats.maxDD;
testWinRate(2) = stats.winRate;
testProfitFactor(2) = stats.profitFactor;
testTrades(2) = stats.trades;

testBuy(2) = sum(position==1);
testHold(2) = sum(position==0);
testSell(2) = sum(position==-1);

%% ---------------- BAGGED TEST ----------------

pred = predTestBag;
actual = Yte;
d = T.Date(idxTestFactor);

threshold = validationThreshold(3);
position = makePosition(pred,threshold);

[ret,equity,stats] = runBacktest( ...
    actual,position,initialCapital,feeRate,annualFactor);

testPredictions{3} = pred;
testActuals{3} = actual;
testDates{3} = d;
testPositions{3} = position;
testEquity{3} = equity;

testRMSE(3) = calcRMSE(actual,pred);
testMAE(3) = calcMAE(actual,pred);
testMSE(3) = mean((actual-pred).^2,'omitnan');

sst = sum((actual-mean(actual)).^2);
sse = sum((actual-pred).^2);

testR2(3) = 1-sse/max(sst,eps);

testDirectional(3) = ...
    mean(sign(actual)==sign(pred))*100;

testPredictionCorrelation(3) = safeCorrelation(actual,pred);

testFinalCapital(3) = stats.finalCapital;
testReturn(3) = ret;
testSharpe(3) = stats.sharpe;
testSortino(3) = stats.sortino;
testMaxDD(3) = stats.maxDD;
testWinRate(3) = stats.winRate;
testProfitFactor(3) = stats.profitFactor;
testTrades(3) = stats.trades;

testBuy(3) = sum(position==1);
testHold(3) = sum(position==0);
testSell(3) = sum(position==-1);

%% ---------------- RANDOM FOREST TEST ----------------

pred = predTestRF;
actual = Yte;
d = T.Date(idxTestFactor);

threshold = validationThreshold(4);
position = makePosition(pred,threshold);

[ret,equity,stats] = runBacktest( ...
    actual,position,initialCapital,feeRate,annualFactor);

testPredictions{4} = pred;
testActuals{4} = actual;
testDates{4} = d;
testPositions{4} = position;
testEquity{4} = equity;

testRMSE(4) = calcRMSE(actual,pred);
testMAE(4) = calcMAE(actual,pred);
testMSE(4) = mean((actual-pred).^2,'omitnan');

sst = sum((actual-mean(actual)).^2);
sse = sum((actual-pred).^2);

testR2(4) = 1-sse/max(sst,eps);

testDirectional(4) = ...
    mean(sign(actual)==sign(pred))*100;

testPredictionCorrelation(4) = safeCorrelation(actual,pred);

testFinalCapital(4) = stats.finalCapital;
testReturn(4) = ret;
testSharpe(4) = stats.sharpe;
testSortino(4) = stats.sortino;
testMaxDD(4) = stats.maxDD;
testWinRate(4) = stats.winRate;
testProfitFactor(4) = stats.profitFactor;
testTrades(4) = stats.trades;

testBuy(4) = sum(position==1);
testHold(4) = sum(position==0);
testSell(4) = sum(position==-1);

%% ---------------- LSBOOST TEST ----------------

pred = predTestLS;
actual = Yte;
d = T.Date(idxTestFactor);

threshold = validationThreshold(5);
position = makePosition(pred,threshold);

[ret,equity,stats] = runBacktest( ...
    actual,position,initialCapital,feeRate,annualFactor);

testPredictions{5} = pred;
testActuals{5} = actual;
testDates{5} = d;
testPositions{5} = position;
testEquity{5} = equity;

testRMSE(5) = calcRMSE(actual,pred);
testMAE(5) = calcMAE(actual,pred);
testMSE(5) = mean((actual-pred).^2,'omitnan');

sst = sum((actual-mean(actual)).^2);
sse = sum((actual-pred).^2);

testR2(5) = 1-sse/max(sst,eps);

testDirectional(5) = ...
    mean(sign(actual)==sign(pred))*100;

testPredictionCorrelation(5) = safeCorrelation(actual,pred);

testFinalCapital(5) = stats.finalCapital;
testReturn(5) = ret;
testSharpe(5) = stats.sharpe;
testSortino(5) = stats.sortino;
testMaxDD(5) = stats.maxDD;
testWinRate(5) = stats.winRate;
testProfitFactor(5) = stats.profitFactor;
testTrades(5) = stats.trades;

testBuy(5) = sum(position==1);
testHold(5) = sum(position==0);
testSell(5) = sum(position==-1);

%% ------------------------------------------------------------
% 26. BUY & HOLD BENCHMARK
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('BUY & HOLD BENCHMARK\n');
fprintf('============================================================\n');

% Use the same test period as the factor models.
%
% Position = +1 throughout.
% Target return is already t -> t+1.

bhActual = target(idxTestFactor);
bhDates = T.Date(idxTestFactor);

bhPosition = ones(size(bhActual));

[bhReturn,bhEquity,bhStats] = runBacktest( ...
    bhActual,bhPosition,initialCapital,feeRate,annualFactor);

fprintf('Buy & Hold final capital : %.2f\n',bhStats.finalCapital);
fprintf('Buy & Hold return        : %.2f%%\n',bhReturn*100);
fprintf('Buy & Hold Sharpe        : %.4f\n',bhStats.sharpe);
fprintf('Buy & Hold max DD        : %.2f%%\n',bhStats.maxDD*100);

%% ------------------------------------------------------------
% 27. RESULTS TABLE
% ------------------------------------------------------------

results = table( ...
    string(modelNames), ...
    validationThreshold, ...
    validationRMSE, ...
    validationMAE, ...
    validationR2, ...
    validationDirectional, ...
    validationSharpe, ...
    validationReturn, ...
    validationMaxDD, ...
    validationTrades, ...
    testRMSE, ...
    testMAE, ...
    testMSE, ...
    testR2, ...
    testDirectional, ...
    testPredictionCorrelation, ...
    testFinalCapital, ...
    testReturn, ...
    testSharpe, ...
    testSortino, ...
    testMaxDD, ...
    testWinRate, ...
    testProfitFactor, ...
    testTrades, ...
    testBuy, ...
    testHold, ...
    testSell, ...
    'VariableNames',{ ...
    'Model', ...
    'Threshold', ...
    'ValidationRMSE', ...
    'ValidationMAE', ...
    'ValidationR2', ...
    'ValidationDirectionalPct', ...
    'ValidationSharpe', ...
    'ValidationReturn', ...
    'ValidationMaxDD', ...
    'ValidationTrades', ...
    'TestRMSE', ...
    'TestMAE', ...
    'TestMSE', ...
    'TestR2', ...
    'TestDirectionalPct', ...
    'TestPredictionCorrelation', ...
    'TestFinalCapital', ...
    'TestReturn', ...
    'TestSharpe', ...
    'TestSortino', ...
    'TestMaxDD', ...
    'TestWinRate', ...
    'TestProfitFactor', ...
    'TestTrades', ...
    'TestBUY', ...
    'TestHOLD', ...
    'TestSELL'});

%% ------------------------------------------------------------
% 28. ADD BUY & HOLD ROW
% ------------------------------------------------------------

bhRow = table( ...
    "Buy & Hold", ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    NaN, ...
    bhStats.finalCapital, ...
    bhReturn, ...
    bhStats.sharpe, ...
    bhStats.sortino, ...
    bhStats.maxDD, ...
    bhStats.winRate, ...
    bhStats.profitFactor, ...
    bhStats.trades, ...
    sum(bhPosition==1), ...
    0, ...
    0, ...
    'VariableNames',results.Properties.VariableNames);

resultsWithBenchmark = [results; bhRow];

%% ------------------------------------------------------------
% 29. SAVE RESULTS
% ------------------------------------------------------------

comparisonFile = fullfile( ...
    resultsTableDir, ...
    'STEP8_UNIFIED_MODEL_COMPARISON.csv');

writetable(resultsWithBenchmark,comparisonFile);

%% Threshold table

thresholdTable = table( ...
    string(modelNames), ...
    validationThreshold, ...
    validationSharpe, ...
    validationReturn, ...
    validationTrades, ...
    'VariableNames',{ ...
    'Model', ...
    'SelectedThreshold', ...
    'ValidationSharpe', ...
    'ValidationReturn', ...
    'ValidationTrades'});

writetable(thresholdTable, ...
    fullfile(resultsTableDir,'STEP8_MODEL_THRESHOLDS.csv'));

%% ------------------------------------------------------------
% 30. SAVE PREDICTIONS
% ------------------------------------------------------------

allDates = unique([ ...
    testDates{1}; ...
    testDates{2}; ...
    testDates{3}; ...
    testDates{4}; ...
    testDates{5}]);

% Since AR and factor models have the same fixed test period,
% create separate tables and outer-align manually.

predictionTable = table();

for m = 1:nModels

    d = testDates{m};
    p = testPredictions{m};
    a = testActuals{m};
    pos = testPositions{m};

    temp = table( ...
        d(:), ...
        a(:), ...
        p(:), ...
        pos(:), ...
        'VariableNames',{ ...
        'Date', ...
        'ActualReturn', ...
        'PredictedReturn', ...
        'Position'});

    modelPrefix = matlab.lang.makeValidName(modelNames{m});

    temp.Properties.VariableNames{2} = ...
        [modelPrefix '_ActualReturn'];

    temp.Properties.VariableNames{3} = ...
        [modelPrefix '_PredictedReturn'];

    temp.Properties.VariableNames{4} = ...
        [modelPrefix '_Position'];

    if isempty(predictionTable)

        predictionTable = temp;

    else

        predictionTable = outerjoin( ...
            predictionTable, ...
            temp, ...
            'Keys','Date', ...
            'MergeKeys',true);

    end

end

predictionFile = fullfile( ...
    resultsTableDir, ...
    'STEP8_MODEL_PREDICTIONS.csv');

writetable(predictionTable,predictionFile);

%% ------------------------------------------------------------
% 31. SAVE MODELS
% ------------------------------------------------------------

modelFile = fullfile( ...
    resultsModelDir, ...
    'STEP8_UNIFIED_MODELS.mat');

save(modelFile, ...
    'modelObjects', ...
    'modelNames', ...
    'factorNames', ...
    'validationThreshold', ...
    'ridgeLambdaGrid', ...
    'trainStart', ...
    'trainEnd', ...
    'valStart', ...
    'valEnd', ...
    'testStart', ...
    'testEnd', ...
    'feeRate', ...
    'initialCapital');

%% ------------------------------------------------------------
% 32. PRINT FINAL RESULTS
% ------------------------------------------------------------

fprintf('\n');
fprintf('============================================================\n');
fprintf(' FINAL MODEL COMPARISON\n');
fprintf('============================================================\n');

fprintf('\n');
fprintf('%-20s %10s %10s %12s %12s %12s\n', ...
    'Model','Test RMSE','Test R2','Return','Sharpe','Max DD');

fprintf('%s\n',repmat('-',1,82));

for i = 1:nModels

    fprintf('%-20s %10.5f %10.4f %11.2f%% %11.4f %11.2f%%\n', ...
        modelNames{i}, ...
        testRMSE(i), ...
        testR2(i), ...
        testReturn(i)*100, ...
        testSharpe(i), ...
        testMaxDD(i)*100);

end

fprintf('%-20s %10s %10s %11.2f%% %11.4f %11.2f%%\n', ...
    'Buy & Hold', ...
    '-', ...
    '-', ...
    bhReturn*100, ...
    bhStats.sharpe, ...
    bhStats.maxDD*100);

%% ------------------------------------------------------------
% 33. MODEL-BY-MODEL DETAILED OUTPUT
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('DETAILED RESULTS\n');
fprintf('============================================================\n');

for i = 1:nModels

    fprintf('\n%s\n',upper(modelNames{i}));
    fprintf('------------------------------------------------------------\n');

    fprintf('Threshold              : %.4f%%\n', ...
        validationThreshold(i)*100);

    fprintf('Validation RMSE        : %.6f\n', ...
        validationRMSE(i));

    fprintf('Validation MAE         : %.6f\n', ...
        validationMAE(i));

    fprintf('Validation R2          : %.6f\n', ...
        validationR2(i));

    fprintf('Validation Directional  : %.2f%%\n', ...
        validationDirectional(i));

    fprintf('Validation Sharpe      : %.4f\n', ...
        validationSharpe(i));

    fprintf('Validation Return      : %.2f%%\n', ...
        validationReturn(i)*100);

    fprintf('Validation Trades      : %d\n', ...
        validationTrades(i));

    fprintf('\n');

    fprintf('Test RMSE              : %.6f\n', ...
        testRMSE(i));

    fprintf('Test MAE               : %.6f\n', ...
        testMAE(i));

    fprintf('Test R2                : %.6f\n', ...
        testR2(i));

    fprintf('Test Directional       : %.2f%%\n', ...
        testDirectional(i));

    fprintf('Prediction Correlation : %.4f\n', ...
        testPredictionCorrelation(i));

    fprintf('Test Final Capital     : %.2f\n', ...
        testFinalCapital(i));

    fprintf('Test Return            : %.2f%%\n', ...
        testReturn(i)*100);

    fprintf('Test Sharpe            : %.4f\n', ...
        testSharpe(i));

    fprintf('Test Sortino           : %.4f\n', ...
        testSortino(i));

    fprintf('Test Max Drawdown      : %.2f%%\n', ...
        testMaxDD(i)*100);

    fprintf('Test Win Rate          : %.2f%%\n', ...
        testWinRate(i)*100);

    fprintf('Test Profit Factor     : %.4f\n', ...
        testProfitFactor(i));

    fprintf('Test Trades            : %d\n', ...
        testTrades(i));

    fprintf('BUY / HOLD / SELL      : %d / %d / %d\n', ...
        testBuy(i), ...
        testHold(i), ...
        testSell(i));

end

%% ------------------------------------------------------------
% 34. COMPARE AGAINST BUY & HOLD
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('MODEL vs BUY & HOLD\n');
fprintf('============================================================\n');

for i = 1:nModels

    returnDifference = testReturn(i)-bhReturn;
    sharpeDifference = testSharpe(i)-bhStats.sharpe;
    ddDifference = testMaxDD(i)-bhStats.maxDD;

    fprintf('\n%s\n',modelNames{i});

    fprintf('Return difference : %.2f percentage points\n', ...
        returnDifference*100);

    fprintf('Sharpe difference : %.4f\n', ...
        sharpeDifference);

    fprintf('Drawdown difference : %.2f percentage points\n', ...
        ddDifference*100);

end

%% ------------------------------------------------------------
% 35. FIGURE 1 - FINAL RETURN COMPARISON
% ------------------------------------------------------------

figure('Name','Step 8 Return Comparison');

returnsPlot = [testReturn(:); bhReturn];

bar(returnsPlot*100);

grid on;

xticks(1:(nModels+1));
xticklabels([modelNames;{'Buy & Hold'}]);

xtickangle(30);

ylabel('Test Return (%)');
title('Step 8 - Test Period Return Comparison');

saveas(gcf, ...
    fullfile(resultsFigureDir, ...
    'STEP8_MODEL_RETURN_COMPARISON.png'));

%% ------------------------------------------------------------
% 36. FIGURE 2 - EQUITY CURVES
% ------------------------------------------------------------

figure('Name','Step 8 Equity Curves');

hold on;

for i = 1:nModels

    plot(testDates{i},testEquity{i}, ...
        'LineWidth',1.2);

end

plot(bhDates,bhEquity, ...
    'LineWidth',1.5);

grid on;

xlabel('Date');
ylabel('Portfolio Value');

title('Step 8 - Test Period Equity Curves');

legend( ...
    [modelNames;{'Buy & Hold'}], ...
    'Location','best');

saveas(gcf, ...
    fullfile(resultsFigureDir, ...
    'STEP8_MODEL_EQUITY_CURVES.png'));

%% ------------------------------------------------------------
% 37. FIGURE 3 - TEST RMSE
% ------------------------------------------------------------

figure('Name','Step 8 Test RMSE');

bar(testRMSE);

grid on;

xticks(1:nModels);
xticklabels(modelNames);
xtickangle(30);

ylabel('RMSE');
title('Step 8 - Test Prediction RMSE');

saveas(gcf, ...
    fullfile(resultsFigureDir, ...
    'STEP8_TEST_RMSE.png'));

%% ------------------------------------------------------------
% 38. FIGURE 4 - TEST SHARPE
% ------------------------------------------------------------

figure('Name','Step 8 Test Sharpe');

bar(testSharpe);

grid on;

xticks(1:nModels);
xticklabels(modelNames);
xtickangle(30);

ylabel('Sharpe Ratio');
title('Step 8 - Test Sharpe Ratio');

saveas(gcf, ...
    fullfile(resultsFigureDir, ...
    'STEP8_TEST_SHARPE.png'));

%% ------------------------------------------------------------
% 39. FINAL MESSAGE
% ------------------------------------------------------------

fprintf('\n');
fprintf('============================================================\n');
fprintf(' STEP 8 COMPLETED\n');
fprintf('============================================================\n');

fprintf('\nSaved files:\n');

fprintf('1. %s\n',comparisonFile);

fprintf('2. %s\n', ...
    fullfile(resultsTableDir,'STEP8_MODEL_THRESHOLDS.csv'));

fprintf('3. %s\n',predictionFile);

fprintf('4. %s\n', ...
    fullfile(resultsTableDir,'STEP8_FACTOR_LIST.csv'));

fprintf('5. %s\n',modelFile);

fprintf('6. %s\n', ...
    fullfile(resultsFigureDir,'STEP8_MODEL_RETURN_COMPARISON.png'));

fprintf('7. %s\n', ...
    fullfile(resultsFigureDir,'STEP8_MODEL_EQUITY_CURVES.png'));

fprintf('8. %s\n', ...
    fullfile(resultsFigureDir,'STEP8_TEST_RMSE.png'));

fprintf('9. %s\n', ...
    fullfile(resultsFigureDir,'STEP8_TEST_SHARPE.png'));

fprintf('\n');
fprintf('IMPORTANT:\n');
fprintf('The test set was not used for model or threshold selection.\n');
fprintf('All models use the same fixed test dates.\n');
fprintf('============================================================\n');


%% ============================================================
% LOCAL FUNCTIONS
% ============================================================

function position = makePosition(prediction,threshold)

    prediction = prediction(:);

    position = zeros(size(prediction));

    position(prediction > threshold) = 1;
    position(prediction < -threshold) = -1;

end


function [totalReturn,equity,stats] = runBacktest( ...
    actualReturn,position,initialCapital,feeRate,annualFactor)

    actualReturn = actualReturn(:);
    position = position(:);

    n = numel(actualReturn);

    if numel(position) ~= n
        error('Position and return vectors must have same length.');
    end

    % Transaction cost based on turnover.
    previousPosition = [0;position(1:end-1)];

    turnover = abs(position-previousPosition);

    transactionCost = feeRate .* turnover;

    % Strategy daily return
    strategyReturn = ...
        position .* actualReturn - transactionCost;

    strategyReturn(~isfinite(strategyReturn)) = 0;

    equity = initialCapital .* ...
        cumprod(1+strategyReturn);

    finalCapital = equity(end);

    totalReturn = finalCapital/initialCapital - 1;

    % Sharpe
    dailyMean = mean(strategyReturn,'omitnan');
    dailyStd = std(strategyReturn,0,'omitnan');

    if dailyStd > 0 && isfinite(dailyStd)
        sharpe = annualFactor*dailyMean/dailyStd;
    else
        sharpe = 0;
    end

    % Downside deviation
    negativeReturns = strategyReturn(strategyReturn < 0);

    if isempty(negativeReturns)

        downsideStd = 0;

    else

        downsideStd = sqrt(mean(negativeReturns.^2));

    end

    if downsideStd > 0
        sortino = annualFactor*dailyMean/downsideStd;
    else
        sortino = 0;
    end

    % Drawdown
    runningPeak = cummax(equity);

    drawdown = equity./runningPeak - 1;

    maxDD = min(drawdown);

    % Winning trading days
    activeDays = strategyReturn(position ~= 0);

    if isempty(activeDays)

        winRate = 0;

    else

        winRate = mean(activeDays > 0);

    end

    % Profit factor
    grossProfit = sum(strategyReturn(strategyReturn > 0));

    grossLoss = -sum(strategyReturn(strategyReturn < 0));

    if grossLoss > 0
        profitFactor = grossProfit/grossLoss;
    elseif grossProfit > 0
        profitFactor = Inf;
    else
        profitFactor = 0;
    end

    % Count completed position changes as trades.
    trades = sum(abs(diff([0;position])) > 0);

    stats = struct();

    stats.finalCapital = finalCapital;
    stats.sharpe = sharpe;
    stats.sortino = sortino;
    stats.maxDD = maxDD;
    stats.winRate = winRate;
    stats.profitFactor = profitFactor;
    stats.trades = trades;

end


function c = safeCorrelation(a,b)

    a = a(:);
    b = b(:);

    valid = isfinite(a) & isfinite(b);

    a = a(valid);
    b = b(valid);

    if numel(a) < 2

        c = NaN;
        return;

    end

    if std(a) == 0 || std(b) == 0

        c = NaN;
        return;

    end

    C = corrcoef(a,b);

    c = C(1,2);

end