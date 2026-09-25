%% STEP 10 - Reduced Factor Experiment + Fair Threshold Stability
% Crypto Sentiment Factor Trading
%
% MODELS:
%   1. AR(5)
%   2. Original Ridge - 47 factors
%   3. Reduced Ridge - correlation-reduced factors
%
% STEP 10B CHANGE:
%   Reduced Ridge previously used the same absolute threshold grid as
%   AR5/Original Ridge. Because Reduced Ridge predictions are much smaller,
%   no validation threshold generated enough trades.
%
%   This version uses a MODEL-SCALE-INDEPENDENT percentile threshold
%   procedure for all models:
%
%       threshold = percentile(abs(validation predictions), P)
%
%   The percentile P is selected using validation Sharpe.
%
%   Therefore each model is evaluated fairly according to its own
%   prediction scale, while:
%       - training remains unchanged
%       - validation is used for threshold selection
%       - test remains completely untouched
%
% IMPORTANT:
%   This is a research/backtesting experiment only.
%
% =========================================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' STEP 10B - REDUCED FACTOR + FAIR THRESHOLD EXPERIMENT\n');
fprintf('============================================================\n\n');

%% ------------------------------------------------------------------------
% CONFIGURATION
% -------------------------------------------------------------------------

projectRoot = fileparts(fileparts(mfilename('fullpath')));

dataDir    = fullfile(projectRoot,'data','processed');
resultDir  = fullfile(projectRoot,'results');

tableDir   = fullfile(resultDir,'tables');
modelDir   = fullfile(resultDir,'models');
figureDir  = fullfile(resultDir,'figures');

if ~exist(tableDir,'dir')
    mkdir(tableDir);
end

if ~exist(modelDir,'dir')
    mkdir(modelDir);
end

if ~exist(figureDir,'dir')
    mkdir(figureDir);
end

% Input dataset
factorFile = fullfile(dataDir,'factor_dataset.csv');

if ~exist(factorFile,'file')
    error('factor_dataset.csv not found:\n%s',factorFile);
end

% Transaction cost
feeRate = 0.001;       % 0.10%

% Random seed
rng(42);

% Correlation reduction threshold
corrThreshold = 0.90;

% Minimum number of validation trades
minValidationTrades = 4;

% -------------------------------------------------------------------------
% FAIR THRESHOLD SETTINGS
% -------------------------------------------------------------------------
%
% Instead of absolute thresholds such as:
%
%       0.001 : 0.0001 : 0.010
%
% we use prediction percentiles.
%
% Example:
%
%   10th percentile -> more signals
%   20th percentile -> fewer signals
%   ...
%
% The threshold is calculated from:
%
%       abs(validation predictions)
%
% ONLY.
%
% The test set is never used.
%
percentileGrid = [ ...
     5
    10
    15
    20
    25
    30
    35
    40
    45
    50
];

fprintf('Correlation threshold : %.2f\n',corrThreshold);
fprintf('Transaction cost       : %.2f%%\n',feeRate*100);
fprintf('Minimum validation trades: %d\n',minValidationTrades);
fprintf('Threshold method       : validation prediction percentiles\n\n');

%% ------------------------------------------------------------------------
% LOAD DATA
% -------------------------------------------------------------------------

fprintf('Loading factor dataset...\n');

T = readtable(factorFile);

fprintf('Rows loaded: %d\n',height(T));
fprintf('Columns loaded: %d\n\n',width(T));

%% ------------------------------------------------------------------------
% FIND DATE COLUMN
% -------------------------------------------------------------------------

dateCandidates = {'Date','date','Time','time','OpenTime','OpenTimeDate'};

dateVar = '';

for i = 1:numel(dateCandidates)

    idx = find(strcmpi(T.Properties.VariableNames,dateCandidates{i}),1);

    if ~isempty(idx)
        dateVar = T.Properties.VariableNames{idx};
        break;
    end

end

if isempty(dateVar)

    % Try first datetime variable
    for i = 1:width(T)

        if isdatetime(T.(T.Properties.VariableNames{i}))
            dateVar = T.Properties.VariableNames{i};
            break;
        end

    end

end

if isempty(dateVar)
    error('Could not identify Date column.');
end

%% ------------------------------------------------------------------------
% ROBUST DATE CONVERSION
% -------------------------------------------------------------------------

if ~isdatetime(T.(dateVar))

    rawDate = T.(dateVar);

    if iscell(rawDate) || isstring(rawDate) || ischar(rawDate)

        try
            T.(dateVar) = datetime(rawDate);

        catch

            try
                T.(dateVar) = datetime(rawDate,'InputFormat','yyyy-MM-dd');

            catch

                error('Could not parse date column: %s',dateVar);

            end

        end

    elseif isnumeric(rawDate)

        % Try Excel serial dates
        T.(dateVar) = datetime(rawDate,'ConvertFrom','excel');

    else

        error('Unsupported date column format.');

    end

end

%% ------------------------------------------------------------------------
% SORT AND REMOVE DUPLICATES
% -------------------------------------------------------------------------

T = sortrows(T,dateVar);

[~,uniqueIdx] = unique(T.(dateVar),'stable');

T = T(uniqueIdx,:);

fprintf('After sorting/removing duplicate dates: %d rows\n\n',height(T));

%% ------------------------------------------------------------------------
% FIND TARGET COLUMN
% -------------------------------------------------------------------------

targetCandidates = { ...
    'TargetReturn1D',...
    'TargetReturn',...
    'Target',...
    'Return1D_Target',...
    'FutureReturn1D'};

targetVar = '';

for i = 1:numel(targetCandidates)

    idx = find(strcmpi(T.Properties.VariableNames,targetCandidates{i}),1);

    if ~isempty(idx)

        targetVar = T.Properties.VariableNames{idx};
        break;

    end

end

if isempty(targetVar)

    error('Could not identify target return column.');

end

fprintf('Target column: %s\n',targetVar);

%% ------------------------------------------------------------------------
% IDENTIFY FACTORS
% -------------------------------------------------------------------------

allVars = T.Properties.VariableNames;

excludeNames = { ...
    dateVar,...
    targetVar,...
    'Open',...
    'High',...
    'Low',...
    'Close',...
    'Volume',...
    'QuoteAssetVolume',...
    'NumberOfTrades',...
    'TakerBuyBaseAssetVolume',...
    'TakerBuyQuoteAssetVolume',...
    'Ignore',...
    'CloseTime',...
    'OpenTime',...
    'LogReturn'};

factorVars = {};

for i = 1:numel(allVars)

    name = allVars{i};

    if any(strcmpi(name,excludeNames))
        continue;
    end

    if isnumeric(T.(name)) || islogical(T.(name))

        factorVars{end+1} = name; %#ok<SAGROW>

    end

end

% Remove accidental target/leakage-like columns
keep = true(size(factorVars));

for i = 1:numel(factorVars)

    n = lower(factorVars{i});

    if contains(n,'target') || ...
       contains(n,'future') || ...
       contains(n,'nextreturn')

        keep(i) = false;

    end

end

factorVars = factorVars(keep);

fprintf('Usable factors found: %d\n',numel(factorVars));

%% ------------------------------------------------------------------------
% CONVERT FACTOR MATRIX
% -------------------------------------------------------------------------

X = zeros(height(T),numel(factorVars));

for i = 1:numel(factorVars)

    x = T.(factorVars{i});

    if islogical(x)
        x = double(x);
    end

    X(:,i) = double(x);

end

Y = double(T.(targetVar));
dates = T.(dateVar);

%% ------------------------------------------------------------------------
% REMOVE INVALID ROWS
% -------------------------------------------------------------------------

validRows = isfinite(Y) & all(isfinite(X),2);

X = X(validRows,:);
Y = Y(validRows);
dates = dates(validRows);

fprintf('Rows after removing NaN/Inf: %d\n\n',numel(Y));

%% ------------------------------------------------------------------------
% FIXED DATE WINDOWS
% -------------------------------------------------------------------------

trainStart = datetime(2018,1,6);
trainEnd   = datetime(2020,4,10);

valStart   = datetime(2020,4,11);
valEnd     = datetime(2020,10,4);

testStart  = datetime(2020,10,5);
testEnd    = datetime(2021,3,31);

trainMask = dates >= trainStart & dates <= trainEnd;
valMask   = dates >= valStart   & dates <= valEnd;
testMask  = dates >= testStart  & dates <= testEnd;

fprintf('============================================================\n');
fprintf(' FIXED PERIODS\n');
fprintf('============================================================\n');

fprintf('Train      : %s -> %s (%d observations)\n',...
    datestr(trainStart),datestr(trainEnd),sum(trainMask));

fprintf('Validation : %s -> %s (%d observations)\n',...
    datestr(valStart),datestr(valEnd),sum(valMask));

fprintf('Test       : %s -> %s (%d observations)\n\n',...
    datestr(testStart),datestr(testEnd),sum(testMask));

%% ------------------------------------------------------------------------
% EXTRACT PERIODS
% -------------------------------------------------------------------------

XTrain = X(trainMask,:);
YTrain = Y(trainMask);
DTrain = dates(trainMask);

XVal = X(valMask,:);
YVal = Y(valMask);
DVal = dates(valMask);

XTest = X(testMask,:);
YTest = Y(testMask);
DTest = dates(testMask);

%% ------------------------------------------------------------------------
% STEP 10 CORRELATION REDUCTION
% TRAINING DATA ONLY
% -------------------------------------------------------------------------

fprintf('============================================================\n');
fprintf(' CORRELATION REDUCTION\n');
fprintf('============================================================\n');

fprintf('Original factors: %d\n',numel(factorVars));

XTrainReduction = XTrain;

remainingFactors = factorVars;

removedFactors = {};

removedPairs = {};

while true

    C = corrcoef(XTrainReduction,'Rows','pairwise');

    C(1:size(C,1)+1:end) = 0;

    [maxAbsCorr,linearIdx] = max(abs(C(:)));

    if isempty(maxAbsCorr) || maxAbsCorr < corrThreshold
        break;
    end

    [r,c] = ind2sub(size(C),linearIdx);

    if r == c
        break;
    end

    corrValue = C(r,c);

    varR = var(XTrainReduction(:,r),'omitnan');
    varC = var(XTrainReduction(:,c),'omitnan');

    % Remove lower-variance factor
    if varR <= varC

        removeIdx = r;
        keepIdx = c;

    else

        removeIdx = c;
        keepIdx = r;

    end

    factorToRemove = remainingFactors{removeIdx};
    factorToKeep   = remainingFactors{keepIdx};

    removedFactors{end+1,1} = factorToRemove; %#ok<SAGROW>

    removedPairs{end+1,1} = sprintf(...
        '%s vs %s (r = %.5f)',...
        factorToKeep,...
        factorToRemove,...
        corrValue); %#ok<SAGROW>

    fprintf('%2d. %s\n',...
        numel(removedFactors),...
        removedPairs{end});

    XTrainReduction(:,removeIdx) = [];
    remainingFactors(removeIdx) = [];

end

fprintf('\n');

fprintf('Factors removed : %d\n',numel(removedFactors));
fprintf('Factors remaining: %d\n\n',numel(remainingFactors));

%% ------------------------------------------------------------------------
% APPLY SAME FACTOR SELECTION TO ALL DATA
% -------------------------------------------------------------------------

reducedIdx = false(1,numel(factorVars));

for i = 1:numel(factorVars)

    if any(strcmp(factorVars{i},remainingFactors))
        reducedIdx(i) = true;
    end

end

XTrainReduced = XTrain(:,reducedIdx);
XValReduced   = XVal(:,reducedIdx);
XTestReduced  = XTest(:,reducedIdx);

%% ------------------------------------------------------------------------
% STANDARDIZATION
% TRAINING DATA ONLY
% -------------------------------------------------------------------------

muOriginal = mean(XTrain,1,'omitnan');
sigmaOriginal = std(XTrain,0,1,'omitnan');

sigmaOriginal(sigmaOriginal == 0 | ~isfinite(sigmaOriginal)) = 1;

XTrainStd = (XTrain - muOriginal) ./ sigmaOriginal;
XValStd   = (XVal   - muOriginal) ./ sigmaOriginal;
XTestStd  = (XTest  - muOriginal) ./ sigmaOriginal;

% Reduced
muReduced = mean(XTrainReduced,1,'omitnan');
sigmaReduced = std(XTrainReduced,0,1,'omitnan');

sigmaReduced(sigmaReduced == 0 | ~isfinite(sigmaReduced)) = 1;

XTrainReducedStd = ...
    (XTrainReduced - muReduced) ./ sigmaReduced;

XValReducedStd = ...
    (XValReduced - muReduced) ./ sigmaReduced;

XTestReducedStd = ...
    (XTestReduced - muReduced) ./ sigmaReduced;

%% ------------------------------------------------------------------------
% AR(5)
% -------------------------------------------------------------------------

fprintf('Training AR(5)...\n');

% Use actual historical returns
arTrainStart = max(1,6);

arTrainMask = DTrain >= trainStart & DTrain <= trainEnd;

% Build AR lags using full chronological return sequence
allDates = dates;
allReturns = Y;

AR = nan(numel(allReturns),5);

for lag = 1:5
    AR((lag+1):end,lag) = allReturns(1:end-lag);
end

ARTrain = AR(trainMask,:);
ARVal   = AR(valMask,:);
ARTest  = AR(testMask,:);

% Remove invalid AR warmup rows
validARTrain = all(isfinite(ARTrain),2) & isfinite(YTrain);
validARVal   = all(isfinite(ARVal),2)   & isfinite(YVal);
validARTest  = all(isfinite(ARTest),2)  & isfinite(YTest);

ARTrain = ARTrain(validARTrain,:);
YARTrain = YTrain(validARTrain);

ARVal = ARVal(validARVal,:);
YARVal = YVal(validARVal);

ARTest = ARTest(validARTest,:);
YARTest = YTest(validARTest);

DARTest = DTest(validARTest);

%% ------------------------------------------------------------------------
% STANDARDIZE AR
% -------------------------------------------------------------------------

muAR = mean(ARTrain,1);
sigmaAR = std(ARTrain,0,1);

sigmaAR(sigmaAR == 0 | ~isfinite(sigmaAR)) = 1;

ARTrainStd = (ARTrain - muAR)./sigmaAR;
ARValStd   = (ARVal   - muAR)./sigmaAR;
ARTestStd  = (ARTest  - muAR)./sigmaAR;

%% ------------------------------------------------------------------------
% MODEL TRAINING
% -------------------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf(' MODEL TRAINING\n');
fprintf('============================================================\n');

% Ridge lambda
lambda = 1e-4;

% -------------------------------------------------------------------------
% AR5
% -------------------------------------------------------------------------

fprintf('AR(5) Ridge...\n');

modelAR = fitrlinear(...
    ARTrainStd,...
    YARTrain,...
    'Learner','leastsquares',...
    'Regularization','ridge',...
    'Lambda',lambda,...
    'Solver','lbfgs');

predARVal = predict(modelAR,ARValStd);
predARTest = predict(modelAR,ARTestStd);

% -------------------------------------------------------------------------
% ORIGINAL RIDGE
% -------------------------------------------------------------------------

fprintf('Original Ridge (%d factors)...\n',size(XTrainStd,2));

modelOriginal = fitrlinear(...
    XTrainStd,...
    YTrain,...
    'Learner','leastsquares',...
    'Regularization','ridge',...
    'Lambda',lambda,...
    'Solver','lbfgs');

predOriginalVal = predict(modelOriginal,XValStd);
predOriginalTest = predict(modelOriginal,XTestStd);

% -------------------------------------------------------------------------
% REDUCED RIDGE
% -------------------------------------------------------------------------

fprintf('Reduced Ridge (%d factors)...\n',size(XTrainReducedStd,2));

modelReduced = fitrlinear(...
    XTrainReducedStd,...
    YTrain,...
    'Learner','leastsquares',...
    'Regularization','ridge',...
    'Lambda',lambda,...
    'Solver','lbfgs');

predReducedVal = predict(modelReduced,XValReducedStd);
predReducedTest = predict(modelReduced,XTestReducedStd);

fprintf('Models trained successfully.\n\n');

%% ------------------------------------------------------------------------
% FAIR THRESHOLD SELECTION FUNCTION
% -------------------------------------------------------------------------
%
% The function below chooses a threshold from the absolute validation
% prediction distribution.
%
% Example:
%
%   percentile = 20
%
% means:
%
%   threshold = 20th percentile of abs(prediction)
%
% This is automatically adapted to the model's prediction scale.
%
% Threshold is selected using validation Sharpe.
%
% TEST DATA IS NOT USED.
% -------------------------------------------------------------------------

fprintf('============================================================\n');
fprintf(' FAIR THRESHOLD SELECTION\n');
fprintf('============================================================\n');

[thresholdAR, bestPctAR, valStatsAR, thresholdTableAR] = ...
    selectFairThreshold(...
        predARVal,...
        YARVal,...
        percentileGrid,...
        minValidationTrades,...
        feeRate);

[thresholdOriginal, bestPctOriginal, valStatsOriginal, thresholdTableOriginal] = ...
    selectFairThreshold(...
        predOriginalVal,...
        YVal,...
        percentileGrid,...
        minValidationTrades,...
        feeRate);

[thresholdReduced, bestPctReduced, valStatsReduced, thresholdTableReduced] = ...
    selectFairThreshold(...
        predReducedVal,...
        YVal,...
        percentileGrid,...
        minValidationTrades,...
        feeRate);

fprintf('\n');

fprintf('AR5:\n');
fprintf('  Percentile : %.0f%%\n',bestPctAR);
fprintf('  Threshold  : %.8f\n',thresholdAR);
fprintf('  Val Sharpe : %.4f\n',valStatsAR.Sharpe);
fprintf('  Val Return : %.2f%%\n',valStatsAR.Return*100);
fprintf('  Trades     : %d\n',valStatsAR.Trades);

fprintf('\nOriginal Ridge:\n');
fprintf('  Percentile : %.0f%%\n',bestPctOriginal);
fprintf('  Threshold  : %.8f\n',thresholdOriginal);
fprintf('  Val Sharpe : %.4f\n',valStatsOriginal.Sharpe);
fprintf('  Val Return : %.2f%%\n',valStatsOriginal.Return*100);
fprintf('  Trades     : %d\n',valStatsOriginal.Trades);

fprintf('\nReduced Ridge:\n');
fprintf('  Percentile : %.0f%%\n',bestPctReduced);
fprintf('  Threshold  : %.8f\n',thresholdReduced);
fprintf('  Val Sharpe : %.4f\n',valStatsReduced.Sharpe);
fprintf('  Val Return : %.2f%%\n',valStatsReduced.Return*100);
fprintf('  Trades     : %d\n\n',valStatsReduced.Trades);

%% ------------------------------------------------------------------------
% FINAL REFIT ON TRAIN + VALIDATION
% -------------------------------------------------------------------------

fprintf('============================================================\n');
fprintf(' FINAL REFIT\n');
fprintf('============================================================\n');

% Original Ridge
XDevOriginal = [XTrain;XVal];
YDev = [YTrain;YVal];

muDevOriginal = mean(XDevOriginal,1,'omitnan');
sigmaDevOriginal = std(XDevOriginal,0,1,'omitnan');

sigmaDevOriginal(...
    sigmaDevOriginal == 0 | ~isfinite(sigmaDevOriginal)) = 1;

XDevOriginalStd = ...
    (XDevOriginal - muDevOriginal) ./ sigmaDevOriginal;

XTestOriginalStd = ...
    (XTest - muDevOriginal) ./ sigmaDevOriginal;

modelOriginalFinal = fitrlinear(...
    XDevOriginalStd,...
    YDev,...
    'Learner','leastsquares',...
    'Regularization','ridge',...
    'Lambda',lambda,...
    'Solver','lbfgs');

predOriginalTestFinal = ...
    predict(modelOriginalFinal,XTestOriginalStd);

% Reduced Ridge
XDevReduced = [XTrainReduced;XValReduced];

muDevReduced = mean(XDevReduced,1,'omitnan');
sigmaDevReduced = std(XDevReduced,0,1,'omitnan');

sigmaDevReduced(...
    sigmaDevReduced == 0 | ~isfinite(sigmaDevReduced)) = 1;

XDevReducedStd = ...
    (XDevReduced - muDevReduced) ./ sigmaDevReduced;

XTestReducedStd = ...
    (XTestReduced - muDevReduced) ./ sigmaDevReduced;

modelReducedFinal = fitrlinear(...
    XDevReducedStd,...
    YDev,...
    'Learner','leastsquares',...
    'Regularization','ridge',...
    'Lambda',lambda,...
    'Solver','lbfgs');

predReducedTestFinal = ...
    predict(modelReducedFinal,XTestReducedStd);

% AR5
ARDev = [ARTrain;ARVal];
YARDev = [YARTrain;YARVal];

muARDev = mean(ARDev,1);
sigmaARDev = std(ARDev,0,1);

sigmaARDev(sigmaARDev == 0 | ~isfinite(sigmaARDev)) = 1;

ARDevStd = (ARDev - muARDev)./sigmaARDev;

ARTestFinalStd = (ARTest - muARDev)./sigmaARDev;

modelARFinal = fitrlinear(...
    ARDevStd,...
    YARDev,...
    'Learner','leastsquares',...
    'Regularization','ridge',...
    'Lambda',lambda,...
    'Solver','lbfgs');

predARTestFinal = predict(modelARFinal,ARTestFinalStd);

%% ------------------------------------------------------------------------
% TEST METRICS
% -------------------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf(' FINAL TEST PREDICTION METRICS\n');
fprintf('============================================================\n');

metricsAR = predictionMetrics(YARTest,predARTestFinal);

metricsOriginal = predictionMetrics(YTest,predOriginalTestFinal);

metricsReduced = predictionMetrics(YTest,predReducedTestFinal);

fprintf('\nAR5:\n');
printPredictionMetrics(metricsAR);

fprintf('\nOriginal Ridge:\n');
printPredictionMetrics(metricsOriginal);

fprintf('\nReduced Ridge:\n');
printPredictionMetrics(metricsReduced);

%% ------------------------------------------------------------------------
% TEST BACKTESTS
% -------------------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf(' TEST BACKTESTS\n');
fprintf('============================================================\n');

[btAR, sigAR] = runBacktest(...
    predARTestFinal,...
    YARTest,...
    thresholdAR,...
    feeRate);

[btOriginal, sigOriginal] = runBacktest(...
    predOriginalTestFinal,...
    YTest,...
    thresholdOriginal,...
    feeRate);

[btReduced, sigReduced] = runBacktest(...
    predReducedTestFinal,...
    YTest,...
    thresholdReduced,...
    feeRate);

%% ------------------------------------------------------------------------
% BUY AND HOLD
% -------------------------------------------------------------------------

[bhEquity,bhReturn,bhSharpe,bhMaxDD] = ...
    buyHoldBacktest(YTest,feeRate);

%% ------------------------------------------------------------------------
% PRINT RESULTS
% -------------------------------------------------------------------------

fprintf('\nAR5 TEST STRATEGY:\n');
printBacktest(btAR);

fprintf('\nOriginal Ridge TEST STRATEGY:\n');
printBacktest(btOriginal);

fprintf('\nReduced Ridge TEST STRATEGY:\n');
printBacktest(btReduced);

fprintf('\nBuy & Hold:\n');
fprintf('  Return      : %.2f%%\n',bhReturn*100);
fprintf('  Sharpe      : %.4f\n',bhSharpe);
fprintf('  Max DD      : %.2f%%\n',bhMaxDD*100);

%% ------------------------------------------------------------------------
% SIGNAL DISTRIBUTION
% -------------------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf(' SIGNAL DISTRIBUTION\n');
fprintf('============================================================\n');

fprintf('\nAR5:\n');
fprintf('  BUY  : %d\n',sum(sigAR == 1));
fprintf('  HOLD : %d\n',sum(sigAR == 0));
fprintf('  SELL : %d\n',sum(sigAR == -1));

fprintf('\nOriginal Ridge:\n');
fprintf('  BUY  : %d\n',sum(sigOriginal == 1));
fprintf('  HOLD : %d\n',sum(sigOriginal == 0));
fprintf('  SELL : %d\n',sum(sigOriginal == -1));

fprintf('\nReduced Ridge:\n');
fprintf('  BUY  : %d\n',sum(sigReduced == 1));
fprintf('  HOLD : %d\n',sum(sigReduced == 0));
fprintf('  SELL : %d\n',sum(sigReduced == -1));

%% ------------------------------------------------------------------------
% CORRELATION BEFORE / AFTER
% -------------------------------------------------------------------------

Cbefore = corrcoef(XTrain,'Rows','pairwise');

Cafter = corrcoef(XTrainReduced,'Rows','pairwise');

Cbefore(1:size(Cbefore,1)+1:end) = 0;
Cafter(1:size(Cafter,1)+1:end) = 0;

maxCorrBefore = max(abs(Cbefore(:)));
maxCorrAfter = max(abs(Cafter(:)));

numPairsBefore = sum(abs(triu(Cbefore,1)) >= corrThreshold,'all');
numPairsAfter  = sum(abs(triu(Cafter,1)) >= corrThreshold,'all');

fprintf('\n============================================================\n');
fprintf(' CORRELATION SUMMARY\n');
fprintf('============================================================\n');

fprintf('Before reduction:\n');
fprintf('  Factors          : %d\n',numel(factorVars));
fprintf('  High corr pairs  : %d\n',numPairsBefore);
fprintf('  Maximum |r|      : %.5f\n',maxCorrBefore);

fprintf('\nAfter reduction:\n');
fprintf('  Factors          : %d\n',numel(remainingFactors));
fprintf('  High corr pairs  : %d\n',numPairsAfter);
fprintf('  Maximum |r|      : %.5f\n',maxCorrAfter);

%% ------------------------------------------------------------------------
% SUMMARY TABLE
% -------------------------------------------------------------------------

Model = {...
    'AR5';
    'OriginalRidge';
    'ReducedRidge';
    'BuyHold'};

TestRMSE = [...
    metricsAR.RMSE;
    metricsOriginal.RMSE;
    metricsReduced.RMSE;
    NaN];

TestMAE = [...
    metricsAR.MAE;
    metricsOriginal.MAE;
    metricsReduced.MAE;
    NaN];

TestR2 = [...
    metricsAR.R2;
    metricsOriginal.R2;
    metricsReduced.R2;
    NaN];

DirectionalAccuracy = [...
    metricsAR.DirectionalAccuracy;
    metricsOriginal.DirectionalAccuracy;
    metricsReduced.DirectionalAccuracy;
    NaN];

Threshold = [...
    thresholdAR;
    thresholdOriginal;
    thresholdReduced;
    NaN];

Percentile = [...
    bestPctAR;
    bestPctOriginal;
    bestPctReduced;
    NaN];

StrategyReturn = [...
    btAR.Return;
    btOriginal.Return;
    btReduced.Return;
    bhReturn];

Sharpe = [...
    btAR.Sharpe;
    btOriginal.Sharpe;
    btReduced.Sharpe;
    bhSharpe];

MaxDrawdown = [...
    btAR.MaxDrawdown;
    btOriginal.MaxDrawdown;
    btReduced.MaxDrawdown;
    bhMaxDD];

FinalEquity = [...
    btAR.FinalEquity;
    btOriginal.FinalEquity;
    btReduced.FinalEquity;
    10000*(1+bhReturn)];

Trades = [...
    btAR.Trades;
    btOriginal.Trades;
    btReduced.Trades;
    1];

summaryTable = table(...
    Model,...
    TestRMSE,...
    TestMAE,...
    TestR2,...
    DirectionalAccuracy,...
    Percentile,...
    Threshold,...
    StrategyReturn,...
    Sharpe,...
    MaxDrawdown,...
    FinalEquity,...
    Trades);

%% ------------------------------------------------------------------------
% THRESHOLD TABLE
% -------------------------------------------------------------------------

thresholdTableAR.Model = repmat("AR5",height(thresholdTableAR),1);
thresholdTableOriginal.Model = repmat("OriginalRidge",height(thresholdTableOriginal),1);
thresholdTableReduced.Model = repmat("ReducedRidge",height(thresholdTableReduced),1);

thresholdTable = [...
    thresholdTableAR;
    thresholdTableOriginal;
    thresholdTableReduced];

%% ------------------------------------------------------------------------
% REDUCED FACTOR LIST
% -------------------------------------------------------------------------

reducedFactorTable = table(...
    (1:numel(remainingFactors))',...
    string(remainingFactors(:)),...
    'VariableNames',{'Index','Factor'});

%% ------------------------------------------------------------------------
% REMOVED FACTOR TABLE
% -------------------------------------------------------------------------

if isempty(removedFactors)

    removedFactorTable = table(...
        zeros(0,1),...
        strings(0,1),...
        'VariableNames',{'Index','RemovedPair'});

else

    removedFactorTable = table(...
        (1:numel(removedFactors))',...
        string(removedPairs),...
        'VariableNames',{'Index','RemovedPair'});

end

%% ------------------------------------------------------------------------
% PREDICTION TABLE
% -------------------------------------------------------------------------

predictionTable = table(...
    DARTest,...
    YARTest,...
    predARTestFinal,...
    predOriginalTestFinal(validARTest),...
    predReducedTestFinal(validARTest),...
    sigAR,...
    sigOriginal,...
    sigReduced,...
    'VariableNames',{...
    'Date',...
    'ActualReturn',...
    'AR5_PredictedReturn',...
    'OriginalRidge_PredictedReturn',...
    'ReducedRidge_PredictedReturn',...
    'AR5_Signal',...
    'OriginalRidge_Signal',...
    'ReducedRidge_Signal'});

%% ------------------------------------------------------------------------
% SIGNAL TABLE
% -------------------------------------------------------------------------

signalTable = table(...
    ["AR5";"OriginalRidge";"ReducedRidge"],...
    [sum(sigAR==1);sum(sigOriginal==1);sum(sigReduced==1)],...
    [sum(sigAR==0);sum(sigOriginal==0);sum(sigReduced==0)],...
    [sum(sigAR==-1);sum(sigOriginal==-1);sum(sigReduced==-1)],...
    'VariableNames',{'Model','BUY','HOLD','SELL'});

%% ------------------------------------------------------------------------
% SAVE CSV FILES
% -------------------------------------------------------------------------

writetable(...
    summaryTable,...
    fullfile(tableDir,'STEP10B_MODEL_COMPARISON.csv'));

writetable(...
    thresholdTable,...
    fullfile(tableDir,'STEP10B_THRESHOLDS.csv'));

writetable(...
    predictionTable,...
    fullfile(tableDir,'STEP10B_PREDICTIONS.csv'));

writetable(...
    reducedFactorTable,...
    fullfile(tableDir,'STEP10B_REDUCED_FACTOR_LIST.csv'));

writetable(...
    removedFactorTable,...
    fullfile(tableDir,'STEP10B_REMOVED_FACTORS.csv'));

writetable(...
    signalTable,...
    fullfile(tableDir,'STEP10B_SIGNAL_DISTRIBUTION.csv'));

%% ------------------------------------------------------------------------
% SAVE CORRELATION MATRICES
% -------------------------------------------------------------------------

beforeTable = array2table(...
    Cbefore,...
    'VariableNames',matlab.lang.makeUniqueStrings(factorVars));

beforeTable.Properties.RowNames = factorVars;

afterTable = array2table(...
    Cafter,...
    'VariableNames',matlab.lang.makeUniqueStrings(remainingFactors));

afterTable.Properties.RowNames = remainingFactors;

writetable(...
    beforeTable,...
    fullfile(tableDir,'STEP10B_TRAIN_CORRELATION_BEFORE.csv'),...
    'WriteRowNames',true);

writetable(...
    afterTable,...
    fullfile(tableDir,'STEP10B_TRAIN_CORRELATION_AFTER.csv'),...
    'WriteRowNames',true);

%% ------------------------------------------------------------------------
% SAVE MODELS
% -------------------------------------------------------------------------

save(...
    fullfile(modelDir,'STEP10B_REDUCED_FACTOR_MODELS.mat'),...
    'modelARFinal',...
    'modelOriginalFinal',...
    'modelReducedFinal',...
    'remainingFactors',...
    'removedFactors',...
    'muARDev',...
    'sigmaARDev',...
    'muDevOriginal',...
    'sigmaDevOriginal',...
    'muDevReduced',...
    'sigmaDevReduced',...
    'thresholdAR',...
    'thresholdOriginal',...
    'thresholdReduced',...
    'bestPctAR',...
    'bestPctOriginal',...
    'bestPctReduced');

%% ------------------------------------------------------------------------
% FIGURE 1 - CORRELATION BEFORE / AFTER
% -------------------------------------------------------------------------

figure('Visible','off');

bar([...
    numel(factorVars),...
    numel(remainingFactors)]);

set(gca,...
    'XTickLabel',{'Before','After'});

ylabel('Number of Factors');
title('Step 10B - Factor Reduction');

grid on;

saveas(...
    gcf,...
    fullfile(figureDir,'STEP10B_FACTOR_REDUCTION.png'));

close;

%% ------------------------------------------------------------------------
% FIGURE 2 - TEST EQUITY CURVES
% -------------------------------------------------------------------------

figure('Visible','off');

plot(btAR.Equity,'LineWidth',1.5);
hold on;

plot(btOriginal.Equity,'LineWidth',1.5);
plot(btReduced.Equity,'LineWidth',1.5);
plot(bhEquity,'LineWidth',1.5);

xlabel('Test Observation');
ylabel('Equity');
title('Step 10B - Test Equity Curves');

legend(...
    'AR5',...
    'Original Ridge',...
    'Reduced Ridge',...
    'Buy & Hold',...
    'Location','best');

grid on;

saveas(...
    gcf,...
    fullfile(figureDir,'STEP10B_TEST_EQUITY_CURVES.png'));

close;

%% ------------------------------------------------------------------------
% FIGURE 3 - RETURN COMPARISON
% -------------------------------------------------------------------------

figure('Visible','off');

returnsPlot = [...
    btAR.Return,...
    btOriginal.Return,...
    btReduced.Return,...
    bhReturn] * 100;

bar(returnsPlot);

set(gca,...
    'XTickLabel',{'AR5','Original Ridge','Reduced Ridge','Buy & Hold'});

ylabel('Return (%)');
title('Step 10B - Test Return Comparison');

grid on;

saveas(...
    gcf,...
    fullfile(figureDir,'STEP10B_TEST_RETURN_COMPARISON.png'));

close;

%% ------------------------------------------------------------------------
% DIAGNOSTIC TEXT
% -------------------------------------------------------------------------

diagnosticFile = fullfile(...
    tableDir,...
    'STEP10B_DIAGNOSTIC_SUMMARY.txt');

fid = fopen(diagnosticFile,'w');

fprintf(fid,'STEP 10B - FAIR THRESHOLD STABILITY EXPERIMENT\n');
fprintf(fid,'============================================================\n\n');

fprintf(fid,'Dataset:\n');
fprintf(fid,'47 original factors reduced using training-only correlation filtering.\n\n');

fprintf(fid,'Correlation threshold: %.2f\n',corrThreshold);
fprintf(fid,'Original factors: %d\n',numel(factorVars));
fprintf(fid,'Reduced factors: %d\n',numel(remainingFactors));
fprintf(fid,'Maximum |r| before: %.5f\n',maxCorrBefore);
fprintf(fid,'Maximum |r| after : %.5f\n\n',maxCorrAfter);

fprintf(fid,'Threshold methodology:\n');
fprintf(fid,'Thresholds are derived from validation prediction percentiles.\n');
fprintf(fid,'The test set is never used for threshold selection.\n');
fprintf(fid,'This avoids penalizing models whose prediction magnitudes are naturally smaller.\n\n');

fprintf(fid,'AR5 selected percentile: %.0f%%\n',bestPctAR);
fprintf(fid,'AR5 threshold: %.8f\n',thresholdAR);

fprintf(fid,'Original Ridge selected percentile: %.0f%%\n',bestPctOriginal);
fprintf(fid,'Original Ridge threshold: %.8f\n',thresholdOriginal);

fprintf(fid,'Reduced Ridge selected percentile: %.0f%%\n',bestPctReduced);
fprintf(fid,'Reduced Ridge threshold: %.8f\n\n',thresholdReduced);

fprintf(fid,'TEST RESULTS\n');
fprintf(fid,'------------------------------------------------------------\n');

fprintf(fid,'AR5 return: %.4f%%\n',btAR.Return*100);
fprintf(fid,'AR5 Sharpe: %.4f\n',btAR.Sharpe);
fprintf(fid,'AR5 max DD: %.4f%%\n\n',btAR.MaxDrawdown*100);

fprintf(fid,'Original Ridge return: %.4f%%\n',btOriginal.Return*100);
fprintf(fid,'Original Ridge Sharpe: %.4f\n',btOriginal.Sharpe);
fprintf(fid,'Original Ridge max DD: %.4f%%\n\n',btOriginal.MaxDrawdown*100);

fprintf(fid,'Reduced Ridge return: %.4f%%\n',btReduced.Return*100);
fprintf(fid,'Reduced Ridge Sharpe: %.4f\n',btReduced.Sharpe);
fprintf(fid,'Reduced Ridge max DD: %.4f%%\n\n',btReduced.MaxDrawdown*100);

fprintf(fid,'Buy & Hold return: %.4f%%\n',bhReturn*100);
fprintf(fid,'Buy & Hold Sharpe: %.4f\n',bhSharpe);
fprintf(fid,'Buy & Hold max DD: %.4f%%\n',bhMaxDD*100);

fclose(fid);

%% ------------------------------------------------------------------------
% FINAL SUMMARY
% -------------------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf(' STEP 10B COMPLETE\n');
fprintf('============================================================\n');

fprintf('\nFactor reduction:\n');
fprintf('  %d -> %d factors\n',numel(factorVars),numel(remainingFactors));
fprintf('  Max |r|: %.5f -> %.5f\n',maxCorrBefore,maxCorrAfter);

fprintf('\nFair threshold selection:\n');
fprintf('  AR5            : %.0f%% percentile, threshold %.8f\n',...
    bestPctAR,thresholdAR);

fprintf('  Original Ridge : %.0f%% percentile, threshold %.8f\n',...
    bestPctOriginal,thresholdOriginal);

fprintf('  Reduced Ridge  : %.0f%% percentile, threshold %.8f\n',...
    bestPctReduced,thresholdReduced);

fprintf('\nTest results:\n');

fprintf('  AR5            : Return %.2f%% | Sharpe %.4f | Trades %d\n',...
    btAR.Return*100,...
    btAR.Sharpe,...
    btAR.Trades);

fprintf('  Original Ridge : Return %.2f%% | Sharpe %.4f | Trades %d\n',...
    btOriginal.Return*100,...
    btOriginal.Sharpe,...
    btOriginal.Trades);

fprintf('  Reduced Ridge  : Return %.2f%% | Sharpe %.4f | Trades %d\n',...
    btReduced.Return*100,...
    btReduced.Sharpe,...
    btReduced.Trades);

fprintf('  Buy & Hold     : Return %.2f%% | Sharpe %.4f\n',...
    bhReturn*100,...
    bhSharpe);

fprintf('\nFiles saved to:\n%s\n',tableDir);
fprintf('%s\n',modelDir);
fprintf('%s\n',figureDir);

fprintf('\n============================================================\n\n');


%% =========================================================================
% LOCAL FUNCTION: SELECT FAIR THRESHOLD
% =========================================================================

function [bestThreshold,bestPercentile,bestStats,resultTable] = ...
    selectFairThreshold(predVal,actualVal,percentileGrid,minTrades,feeRate)

    predVal = predVal(:);
    actualVal = actualVal(:);

    valid = isfinite(predVal) & isfinite(actualVal);

    predVal = predVal(valid);
    actualVal = actualVal(valid);

    n = numel(predVal);

    numCandidates = numel(percentileGrid);

    Percentile = nan(numCandidates,1);
    Threshold = nan(numCandidates,1);
    Trades = zeros(numCandidates,1);
    Return = nan(numCandidates,1);
    Sharpe = nan(numCandidates,1);
    MaxDrawdown = nan(numCandidates,1);

    for i = 1:numCandidates

        p = percentileGrid(i);

        absPred = abs(predVal);

        threshold = prctile(absPred,p);

        % Avoid zero threshold
        if threshold <= 0
            positiveAbs = absPred(absPred > 0);

            if isempty(positiveAbs)
                threshold = eps;
            else
                threshold = min(positiveAbs);
            end
        end

        signal = zeros(n,1);

        signal(predVal > threshold) = 1;
        signal(predVal < -threshold) = -1;

        bt = runBacktest(...
            predVal,...
            actualVal,...
            threshold,...
            feeRate);

        Percentile(i) = p;
        Threshold(i) = threshold;
        Trades(i) = bt.Trades;
        Return(i) = bt.Return;
        Sharpe(i) = bt.Sharpe;
        MaxDrawdown(i) = bt.MaxDrawdown;

    end

    % Only thresholds satisfying minimum validation trades
    validCandidates = Trades >= minTrades & isfinite(Sharpe);

    if ~any(validCandidates)

        % FALLBACK:
        %
        % Instead of returning NaN and creating an invalid no-trade model,
        % select the percentile producing the largest number of trades.
        %
        % This fallback is still validation-only.

        validCandidates = isfinite(Trades);

        maxTrades = max(Trades(validCandidates));

        candidateIdx = find(...
            validCandidates & ...
            Trades == maxTrades);

        if isempty(candidateIdx)

            error('Could not select a validation threshold.');

        end

        % Among maximum-trade candidates choose highest Sharpe if possible
        candidateSharpe = Sharpe(candidateIdx);

        if any(isfinite(candidateSharpe))

            [~,localIdx] = max(candidateSharpe);
            bestIdx = candidateIdx(localIdx);

        else

            bestIdx = candidateIdx(1);

        end

    else

        candidateIdx = find(validCandidates);

        candidateSharpe = Sharpe(candidateIdx);

        bestSharpe = max(candidateSharpe);

        % Tie-break using higher validation return
        tied = candidateIdx(...
            abs(Sharpe(candidateIdx)-bestSharpe) < 1e-12);

        if numel(tied) > 1

            [~,idxReturn] = max(Return(tied));
            bestIdx = tied(idxReturn);

        else

            bestIdx = tied(1);

        end

    end

    bestThreshold = Threshold(bestIdx);
    bestPercentile = Percentile(bestIdx);

    bestStats = struct();

    bestStats.Return = Return(bestIdx);
    bestStats.Sharpe = Sharpe(bestIdx);
    bestStats.MaxDrawdown = MaxDrawdown(bestIdx);
    bestStats.Trades = Trades(bestIdx);

    resultTable = table(...
        Percentile,...
        Threshold,...
        Trades,...
        Return,...
        Sharpe,...
        MaxDrawdown);

end


%% =========================================================================
% LOCAL FUNCTION: PREDICTION METRICS
% =========================================================================

function M = predictionMetrics(actual,predicted)

actual = actual(:);
predicted = predicted(:);

valid = isfinite(actual) & isfinite(predicted);

actual = actual(valid);
predicted = predicted(valid);

err = predicted - actual;

M = struct();

M.RMSE = sqrt(mean(err.^2));
M.MAE = mean(abs(err));
M.MSE = mean(err.^2);

ssRes = sum((actual-predicted).^2);
ssTot = sum((actual-mean(actual)).^2);

if ssTot > 0
    M.R2 = 1 - ssRes/ssTot;
else
    M.R2 = NaN;
end

M.Bias = mean(predicted-actual);

if std(actual) > 0 && std(predicted) > 0
    M.Correlation = corr(actual,predicted);
else
    M.Correlation = NaN;
end

M.DirectionalAccuracy = ...
    mean(sign(actual)==sign(predicted));

M.PredictionMean = mean(predicted);
M.PredictionStd = std(predicted);

end


%% =========================================================================
% LOCAL FUNCTION: PRINT PREDICTION METRICS
% =========================================================================

function printPredictionMetrics(M)

fprintf('  RMSE                 : %.6f\n',M.RMSE);
fprintf('  MAE                  : %.6f\n',M.MAE);
fprintf('  MSE                  : %.6f\n',M.MSE);
fprintf('  R2                   : %.6f\n',M.R2);
fprintf('  Bias                 : %.6f\n',M.Bias);
fprintf('  Correlation          : %.6f\n',M.Correlation);
fprintf('  Directional Accuracy : %.2f%%\n',M.DirectionalAccuracy*100);
fprintf('  Prediction Mean      : %.8f\n',M.PredictionMean);
fprintf('  Prediction Std       : %.8f\n',M.PredictionStd);

end


%% =========================================================================
% LOCAL FUNCTION: BACKTEST
% =========================================================================

function [stats,signal] = runBacktest(predictedReturn,actualReturn,threshold,feeRate)

predictedReturn = predictedReturn(:);
actualReturn = actualReturn(:);

n = min(numel(predictedReturn),numel(actualReturn));

predictedReturn = predictedReturn(1:n);
actualReturn = actualReturn(1:n);

if ~isfinite(threshold)
    threshold = Inf;
end

signal = zeros(n,1);

signal(predictedReturn > threshold) = 1;
signal(predictedReturn < -threshold) = -1;

position = signal;

% Position change
previousPosition = zeros(n,1);

if n > 1
    previousPosition(2:end) = position(1:end-1);
end

turnover = abs(position - previousPosition);

% Transaction cost
cost = feeRate .* turnover;

strategyReturn = position .* actualReturn - cost;

% Equity
equity = zeros(n,1);
equity(1) = 10000;

for t = 2:n

    equity(t) = equity(t-1) * (1 + strategyReturn(t));

end

% Handle first observation
if n >= 1
    equity(1) = 10000 * (1 + strategyReturn(1));
end

% Prevent invalid equity
equity(~isfinite(equity)) = 10000;

finalEquity = equity(end);

totalReturn = finalEquity/10000 - 1;

% Sharpe
dailyMean = mean(strategyReturn);
dailyStd = std(strategyReturn);

if dailyStd > 0
    sharpe = sqrt(365) * dailyMean/dailyStd;
else
    sharpe = NaN;
end

% Sortino
negativeReturns = strategyReturn(strategyReturn < 0);

if isempty(negativeReturns)

    sortino = NaN;

else

    downsideDeviation = sqrt(mean(negativeReturns.^2));

    if downsideDeviation > 0
        sortino = sqrt(365)*dailyMean/downsideDeviation;
    else
        sortino = NaN;
    end

end

% Drawdown
runningMax = cummax(equity);

drawdown = equity./runningMax - 1;

maxDrawdown = min(drawdown);

% Win rate
active = position ~= 0;

if any(active)

    wins = strategyReturn(active) > 0;

    winRate = mean(wins);

else

    winRate = NaN;

end

% Profit factor
grossProfit = sum(strategyReturn(strategyReturn > 0));
grossLoss = -sum(strategyReturn(strategyReturn < 0));

if grossLoss > 0
    profitFactor = grossProfit/grossLoss;
else
    profitFactor = NaN;
end

% Trades
trades = sum(turnover > 0);

stats = struct();

stats.Return = totalReturn;
stats.Sharpe = sharpe;
stats.Sortino = sortino;
stats.MaxDrawdown = maxDrawdown;
stats.FinalEquity = finalEquity;
stats.WinRate = winRate;
stats.ProfitFactor = profitFactor;
stats.Trades = trades;
stats.Equity = equity;
stats.StrategyReturn = strategyReturn;
stats.Position = position;
stats.Turnover = turnover;
stats.Cost = cost;

end


%% =========================================================================
% LOCAL FUNCTION: PRINT BACKTEST
% =========================================================================

function printBacktest(stats)

fprintf('  Final Equity : $%.2f\n',stats.FinalEquity);
fprintf('  Return       : %.4f%%\n',stats.Return*100);
fprintf('  Sharpe       : %.4f\n',stats.Sharpe);
fprintf('  Sortino      : %.4f\n',stats.Sortino);
fprintf('  Max DD       : %.4f%%\n',stats.MaxDrawdown*100);
fprintf('  Win Rate     : %.2f%%\n',stats.WinRate*100);
fprintf('  Profit Factor: %.4f\n',stats.ProfitFactor);
fprintf('  Trades       : %d\n',stats.Trades);

end


%% =========================================================================
% LOCAL FUNCTION: BUY AND HOLD
% =========================================================================

function [equity,totalReturn,sharpe,maxDrawdown] = ...
    buyHoldBacktest(actualReturn,feeRate)

actualReturn = actualReturn(:);

n = numel(actualReturn);

position = ones(n,1);

previousPosition = zeros(n,1);

if n > 1
    previousPosition(2:end) = 1;
end

turnover = abs(position-previousPosition);

cost = feeRate .* turnover;

dailyReturn = actualReturn - cost;

equity = zeros(n,1);

if n >= 1
    equity(1) = 10000*(1+dailyReturn(1));
end

for t = 2:n

    equity(t) = equity(t-1)*(1+dailyReturn(t));

end

totalReturn = equity(end)/10000 - 1;

dailyMean = mean(dailyReturn);
dailyStd = std(dailyReturn);

if dailyStd > 0
    sharpe = sqrt(365)*dailyMean/dailyStd;
else
    sharpe = NaN;
end

runningMax = cummax(equity);

drawdown = equity./runningMax - 1;

maxDrawdown = min(drawdown);

end