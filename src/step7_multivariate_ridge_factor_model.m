%% ============================================================
% STEP 7: MULTIVARIATE RIDGE FACTOR MODEL - FINAL CORRECTED
%
% Crypto Sentiment Factor Trading
%
% PURPOSE:
% Multivariate Ridge regression using the engineered factors
% already present in factor_dataset.csv.
%
% IMPORTANT:
% This version keeps the SAME FINAL TEST PERIOD used by the
% rigorous Step 5 AR(5) baseline:
%
% TEST:
% 05-Oct-2020 -> 31-Mar-2021
%
% This makes the comparison with AR(5) much more meaningful.
%
% METHODOLOGY:
%   1. Load factor_dataset.csv
%   2. Detect engineered predictors
%   3. Remove leakage variables
%   4. Keep fixed chronological date boundaries
%   5. Train-only standardization
%   6. Select Ridge lambda using validation RMSE
%   7. Select threshold using validation Sharpe
%   8. Require actual validation trades
%   9. Refit on train + validation
%  10. Evaluate untouched test
%  11. Backtest
%  12. Compare against Buy & Hold
%
% This is for academic research/backtesting.
% It is NOT a live trading system.
%% ============================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' [STEP 7] MULTIVARIATE RIDGE FACTOR MODEL\n');
fprintf('============================================================\n\n');

%% ============================================================
% CONFIGURATION
%% ============================================================

initialCapital = 10000;
feeRate = 0.001;                 % 0.10%

% Fixed dates matching rigorous Step 5
trainStartDate = datetime(2018,1,6);
validationStartDate = datetime(2020,4,11);
testStartDate = datetime(2020,10,5);
testEndDate = datetime(2021,3,31);

% Ridge lambda grid
lambdaGrid = logspace(-6,2,17);

% IMPORTANT:
% Ridge predictions are much smaller than the previous AR model.
% Therefore use a finer threshold grid.
thresholdGrid = 0.0001:0.0001:0.0100;

% Minimum number of validation trades
minValidationTrades = 4;

%% ============================================================
% PROJECT PATHS
%% ============================================================

projectRoot = fileparts(fileparts(mfilename('fullpath')));

dataDir = fullfile(projectRoot,'data','processed');
resultsDir = fullfile(projectRoot,'results');

figuresDir = fullfile(resultsDir,'figures');
tablesDir = fullfile(resultsDir,'tables');
modelsDir = fullfile(resultsDir,'models');

if ~exist(figuresDir,'dir')
    mkdir(figuresDir);
end

if ~exist(tablesDir,'dir')
    mkdir(tablesDir);
end

if ~exist(modelsDir,'dir')
    mkdir(modelsDir);
end

dataFile = fullfile(dataDir,'factor_dataset.csv');

if ~isfile(dataFile)

    dataFile = fullfile(dataDir,'merged_dataset.csv');

    if ~isfile(dataFile)
        error(['Could not find factor_dataset.csv or merged_dataset.csv.\n' ...
               'Expected directory:\n%s'],dataDir);
    end

end

fprintf('Data file:\n%s\n\n',dataFile);

%% ============================================================
% LOAD DATA
%% ============================================================

T = readtable(dataFile);

fprintf('Original rows    : %d\n',height(T));
fprintf('Original columns : %d\n\n',width(T));

%% ============================================================
% IDENTIFY DATE COLUMN
%% ============================================================

names = T.Properties.VariableNames;

dateCandidates = { ...
    'Date', ...
    'date', ...
    'Time', ...
    'time', ...
    'OpenTime', ...
    'OpenTimeDate'};

dateVar = '';

for i = 1:numel(dateCandidates)

    idx = find(strcmpi(names,dateCandidates{i}),1);

    if ~isempty(idx)

        dateVar = names{idx};

        break;

    end

end

if isempty(dateVar)

    for i = 1:numel(names)

        lowName = lower(names{i});

        if contains(lowName,'date') || contains(lowName,'time')

            dateVar = names{i};

            break;

        end

    end

end

if isempty(dateVar)
    error('Could not identify date column.');
end

fprintf('Date column : %s\n',dateVar);

%% ============================================================
% PARSE DATE
%% ============================================================

rawDate = T.(dateVar);

if isdatetime(rawDate)

    dates = rawDate;

elseif isnumeric(rawDate)

    dates = datetime(rawDate,'ConvertFrom','datenum');

elseif isstring(rawDate) || ischar(rawDate) || iscell(rawDate)

    try

        dates = datetime(rawDate);

    catch

        dates = datetime(string(rawDate));

    end

else

    error('Unsupported date type.');

end

dates = dates(:);

%% ============================================================
% SORT CHRONOLOGICALLY
%% ============================================================

[dates,sortIdx] = sort(dates);

T = T(sortIdx,:);

%% ============================================================
% REMOVE DUPLICATE DATES
%% ============================================================

[uniqueDates,uniqueIdx] = unique(dates,'stable');

if numel(uniqueDates) < numel(dates)

    fprintf('Duplicate dates removed: %d\n', ...
        numel(dates)-numel(uniqueDates));

    T = T(uniqueIdx,:);

    dates = uniqueDates;

end

fprintf('Date range : %s -> %s\n', ...
    datestr(dates(1)),datestr(dates(end)));

fprintf('Rows after sorting : %d\n\n',height(T));

%% ============================================================
% IDENTIFY TARGET
%% ============================================================

targetCandidates = { ...
    'TargetReturn1D', ...
    'TargetReturn', ...
    'NextDayReturn', ...
    'FutureReturn', ...
    'Target', ...
    'target'};

targetVar = '';

for i = 1:numel(targetCandidates)

    idx = find(strcmpi(T.Properties.VariableNames, ...
                       targetCandidates{i}),1);

    if ~isempty(idx)

        targetVar = T.Properties.VariableNames{idx};

        break;

    end

end

%% ============================================================
% CONSTRUCT TARGET IF NECESSARY
%% ============================================================

if isempty(targetVar)

    closeCandidates = { ...
        'Close', ...
        'close', ...
        'BTC_Close', ...
        'Price'};

    closeVar = '';

    for i = 1:numel(closeCandidates)

        idx = find(strcmpi(T.Properties.VariableNames, ...
                           closeCandidates{i}),1);

        if ~isempty(idx)

            closeVar = T.Properties.VariableNames{idx};

            break;

        end

    end

    if isempty(closeVar)
        error('No target or Close column found.');
    end

    closePrice = convertToNumeric(T.(closeVar));

    target = nan(height(T),1);

    for t = 1:height(T)-1

        if isfinite(closePrice(t)) && ...
           isfinite(closePrice(t+1)) && ...
           closePrice(t) ~= 0

            target(t) = ...
                closePrice(t+1)/closePrice(t)-1;

        end

    end

    T.Step7Target = target;

    targetVar = 'Step7Target';

else

    T.Step7Target = ...
        convertToNumeric(T.(targetVar));

end

fprintf('Existing target detected: %s\n\n',targetVar);

%% ============================================================
% DEFINE PREFERRED FACTORS
%
% These are matched to the actual dataset names.
%% ============================================================

preferredFactors = { ...
    'Return', ...
    'LogReturn', ...
    'ReturnLag1', ...
    'ReturnLag2', ...
    'ReturnLag3', ...
    'ReturnLag5', ...
    'ReturnLag7', ...
    'Momentum3', ...
    'Momentum5', ...
    'Momentum7', ...
    'Momentum14', ...
    'Momentum30', ...
    'SMA5', ...
    'SMA10', ...
    'SMA20', ...
    'SMA50', ...
    'EMA12', ...
    'EMA26', ...
    'EMA12Distance', ...
    'EMA26Distance', ...
    'SMA20Distance', ...
    'SMA50Distance', ...
    'RSI14', ...
    'MACD', ...
    'MACDSignal', ...
    'MACDHistogram', ...
    'Volatility5', ...
    'Volatility10', ...
    'Volatility20', ...
    'Volatility30', ...
    'ATR14', ...
    'ATR14Normalized', ...
    'DailyRange', ...
    'VolumeChange', ...
    'VolumeRelative', ...
    'SentimentLag1', ...
    'SentimentLag3', ...
    'SentimentLag7', ...
    'SentimentMA3', ...
    'SentimentMA7', ...
    'SentimentMA14', ...
    'SentimentChange', ...
    'SentimentReturnInteraction', ...
    'SentimentMomentumInteraction'};

%% ============================================================
% LEAKAGE / RAW VARIABLE EXCLUSIONS
%% ============================================================

excludedNames = { ...
    dateVar, ...
    targetVar, ...
    'Step7Target', ...
    'TargetDirection', ...
    'SentimentClass', ...
    'Open', ...
    'High', ...
    'Low', ...
    'Close', ...
    'Volume', ...
    'NumberOfTrades', ...
    'SampleSize', ...
    'Open time', ...
    'Close time', ...
    'Quote asset volume', ...
    'Taker buy base asset volume', ...
    'Taker buy quote asset volume', ...
    'Ignore'};

%% ============================================================
% FACTOR DETECTION
%% ============================================================

selectedFactors = {};

fprintf('============================================================\n');
fprintf(' FACTOR DETECTION\n');
fprintf('============================================================\n\n');

for i = 1:numel(preferredFactors)

    idx = find(strcmpi(T.Properties.VariableNames, ...
                       preferredFactors{i}),1);

    if ~isempty(idx)

        selectedFactors{end+1} = ...
            T.Properties.VariableNames{idx}; %#ok<SAGROW>

        fprintf('%-35s -> FOUND\n',preferredFactors{i});

    else

        fprintf('%-35s -> NOT FOUND\n',preferredFactors{i});

    end

end

%% ============================================================
% ADD OTHER VALID NUMERIC ENGINEERED FACTORS
%
% This catches useful engineered columns not listed above.
%% ============================================================

for i = 1:numel(T.Properties.VariableNames)

    varName = T.Properties.VariableNames{i};

    if any(strcmpi(selectedFactors,varName))
        continue;
    end

    if any(strcmpi(excludedNames,varName))
        continue;
    end

    lowName = lower(varName);

    % Remove anything that could contain future information
    leakage = ...
        contains(lowName,'target') || ...
        contains(lowName,'future') || ...
        contains(lowName,'forward') || ...
        contains(lowName,'nextday') || ...
        contains(lowName,'label') || ...
        contains(lowName,'class');

    if leakage
        continue;
    end

    value = T.(varName);

    if isnumeric(value) || islogical(value)

        numericValue = double(value);

        if any(isfinite(numericValue(:)))

            selectedFactors{end+1} = varName; %#ok<SAGROW>

        end

    end

end

%% ============================================================
% REMOVE DUPLICATES
%% ============================================================

selectedFactors = unique(selectedFactors,'stable');

%% ============================================================
% DISPLAY FINAL FACTORS
%% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' FINAL FACTOR SET\n');
fprintf('============================================================\n\n');

fprintf('Number of factors: %d\n\n', ...
    numel(selectedFactors));

for i = 1:numel(selectedFactors)

    fprintf('%2d. %s\n', ...
        i,selectedFactors{i});

end

fprintf('\n');

if numel(selectedFactors) < 5
    error('Too few usable factors were found.');
end

%% ============================================================
% BUILD FEATURE MATRIX
%% ============================================================

X = nan(height(T),numel(selectedFactors));

for j = 1:numel(selectedFactors)

    X(:,j) = convertToNumeric( ...
        T.(selectedFactors{j}));

end

y = convertToNumeric(T.Step7Target);

%% ============================================================
% VALID ROW MASK
%% ============================================================

validRows = ...
    isfinite(y) & ...
    all(isfinite(X),2);

fprintf('Rows before feature cleaning : %d\n',height(T));
fprintf('Rows after feature cleaning  : %d\n',sum(validRows));
fprintf('Rows removed                 : %d\n\n',sum(~validRows));

X = X(validRows,:);
y = y(validRows);
datesClean = dates(validRows);

%% ============================================================
% FIXED DATE-BASED SPLIT
%
% This is the important correction.
%
% We DO NOT split 70/15/15 after feature cleaning.
% Instead, we preserve the exact research periods.
%% ============================================================

trainMask = ...
    datesClean >= trainStartDate & ...
    datesClean < validationStartDate;

validationMask = ...
    datesClean >= validationStartDate & ...
    datesClean < testStartDate;

testMask = ...
    datesClean >= testStartDate & ...
    datesClean <= testEndDate;

XTrain = X(trainMask,:);
yTrain = y(trainMask);
datesTrain = datesClean(trainMask);

XVal = X(validationMask,:);
yVal = y(validationMask);
datesVal = datesClean(validationMask);

XTest = X(testMask,:);
yTest = y(testMask);
datesTest = datesClean(testMask);

fprintf('============================================================\n');
fprintf(' FIXED CHRONOLOGICAL SPLIT\n');
fprintf('============================================================\n\n');

fprintf('Train:\n');
fprintf('  Observations : %d\n',numel(yTrain));
fprintf('  Start        : %s\n',datestr(datesTrain(1)));
fprintf('  End          : %s\n\n',datestr(datesTrain(end)));

fprintf('Validation:\n');
fprintf('  Observations : %d\n',numel(yVal));
fprintf('  Start        : %s\n',datestr(datesVal(1)));
fprintf('  End          : %s\n\n',datestr(datesVal(end)));

fprintf('Test:\n');
fprintf('  Observations : %d\n',numel(yTest));
fprintf('  Start        : %s\n',datestr(datesTest(1)));
fprintf('  End          : %s\n\n',datestr(datesTest(end)));

%% ============================================================
% VERIFY TEST PERIOD
%% ============================================================

if datesTest(1) ~= testStartDate

    warning(['The first available test date is %s instead of ' ...
             'the requested %s because that row did not contain ' ...
             'valid factor values.'], ...
             datestr(datesTest(1)), ...
             datestr(testStartDate));

end

%% ============================================================
% TRAIN-ONLY STANDARDIZATION
%% ============================================================

muTrain = mean(XTrain,1);
sigmaTrain = std(XTrain,0,1);

sigmaTrain(sigmaTrain < eps) = 1;

XTrainZ = ...
    (XTrain-muTrain)./sigmaTrain;

XValZ = ...
    (XVal-muTrain)./sigmaTrain;

XTestZ = ...
    (XTest-muTrain)./sigmaTrain;

%% ============================================================
% RIDGE LAMBDA SELECTION
%% ============================================================

fprintf('============================================================\n');
fprintf(' RIDGE LAMBDA SELECTION\n');
fprintf('============================================================\n\n');

lambdaResults = nan(numel(lambdaGrid),5);

bestLambda = NaN;
bestValRMSE = Inf;

for k = 1:numel(lambdaGrid)

    lambda = lambdaGrid(k);

    model = fitrlinear( ...
        XTrainZ, ...
        yTrain, ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda',lambda, ...
        'Solver','lbfgs', ...
        'FitBias',true);

    predValLambda = predict(model,XValZ);

    currentRMSE = sqrt( ...
        mean((predValLambda-yVal).^2));

    currentMAE = mean( ...
        abs(predValLambda-yVal));

    SSres = sum( ...
        (yVal-predValLambda).^2);

    SStot = sum( ...
        (yVal-mean(yVal)).^2);

    if SStot > 0
        currentR2 = 1-SSres/SStot;
    else
        currentR2 = NaN;
    end

    currentDirectional = ...
        mean(sign(predValLambda)==sign(yVal))*100;

    lambdaResults(k,:) = [ ...
        lambda, ...
        currentRMSE, ...
        currentMAE, ...
        currentR2, ...
        currentDirectional];

    fprintf(['Lambda = %-10.6g | RMSE = %.6f | ' ...
             'MAE = %.6f | R2 = %.6f | Direction = %.2f%%\n'], ...
        lambda, ...
        currentRMSE, ...
        currentMAE, ...
        currentR2, ...
        currentDirectional);

    if currentRMSE < bestValRMSE

        bestValRMSE = currentRMSE;
        bestLambda = lambda;

    end

end

fprintf('\n');
fprintf('Selected Ridge lambda : %.6g\n',bestLambda);
fprintf('Validation RMSE       : %.6f\n\n',bestValRMSE);

%% ============================================================
% TRAIN DEVELOPMENT MODEL
%% ============================================================

developmentModel = fitrlinear( ...
    XTrainZ, ...
    yTrain, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda',bestLambda, ...
    'Solver','lbfgs', ...
    'FitBias',true);

predTrain = predict(developmentModel,XTrainZ);
predVal = predict(developmentModel,XValZ);

%% ============================================================
% DEVELOPMENT METRICS
%% ============================================================

trainRMSE = sqrt(mean((predTrain-yTrain).^2));
trainMAE = mean(abs(predTrain-yTrain));

valRMSE = sqrt(mean((predVal-yVal).^2));
valMAE = mean(abs(predVal-yVal));

SSres = sum((yVal-predVal).^2);
SStot = sum((yVal-mean(yVal)).^2);

if SStot > 0
    valR2 = 1-SSres/SStot;
else
    valR2 = NaN;
end

valDirectional = ...
    mean(sign(predVal)==sign(yVal))*100;

%% ============================================================
% SHOW PREDICTION RANGE
%% ============================================================

fprintf('Validation prediction range:\n');
fprintf('  Minimum : %.6f\n',min(predVal));
fprintf('  Maximum : %.6f\n',max(predVal));
fprintf('  Mean    : %.6f\n',mean(predVal));
fprintf('  Std     : %.6f\n\n',std(predVal));

%% ============================================================
% VALIDATION THRESHOLD SELECTION
%
% IMPORTANT:
% We search thresholds from 0.01% to 1%.
%
% Zero-trade thresholds cannot win.
%% ============================================================

fprintf('============================================================\n');
fprintf(' VALIDATION THRESHOLD SELECTION\n');
fprintf('============================================================\n\n');

fprintf('Minimum validation trades required: %d\n\n', ...
    minValidationTrades);

thresholdResults = nan(numel(thresholdGrid),7);

bestThreshold = NaN;
bestValSharpe = -Inf;
bestValReturn = -Inf;

for k = 1:numel(thresholdGrid)

    threshold = thresholdGrid(k);

    positionsVal = zeros(size(predVal));

    positionsVal(predVal > threshold) = 1;
    positionsVal(predVal < -threshold) = -1;

    [valStats,~,~] = runBacktest( ...
        yVal, ...
        positionsVal, ...
        initialCapital, ...
        feeRate);

    thresholdResults(k,:) = [ ...
        threshold, ...
        valStats.return, ...
        valStats.sharpe, ...
        valStats.sortino, ...
        valStats.maxDrawdown, ...
        valStats.winRate, ...
        valStats.trades];

    if valStats.trades < minValidationTrades

        continue;

    end

    fprintf(['Threshold = %5.2f%% | Return = %8.2f%% | ' ...
             'Sharpe = %7.4f | MaxDD = %7.2f%% | ' ...
             'Trades = %d | USED\n'], ...
        threshold*100, ...
        valStats.return*100, ...
        valStats.sharpe, ...
        valStats.maxDrawdown*100, ...
        valStats.trades);

    if valStats.sharpe > bestValSharpe + 1e-12 || ...
       (abs(valStats.sharpe-bestValSharpe)<=1e-12 && ...
        valStats.return > bestValReturn)

        bestValSharpe = valStats.sharpe;
        bestValReturn = valStats.return;
        bestThreshold = threshold;

    end

end

%% ============================================================
% FALLBACK
%
% If no threshold generated >= 4 trades, do NOT pretend that
% the strategy has a validated trading threshold.
%
% Instead use the threshold with the highest number of trades,
% then highest validation return.
%% ============================================================

if isnan(bestThreshold)

    fprintf('\n');
    fprintf('WARNING: No threshold produced at least %d trades.\n', ...
        minValidationTrades);

    fprintf(['The Ridge predictions are too concentrated around ' ...
             'zero for the requested minimum-trade rule.\n']);

    fprintf(['Selecting the threshold with the largest number ' ...
             'of validation trades for diagnostic purposes.\n\n']);

    tradeCounts = thresholdResults(:,7);

    maxTrades = max(tradeCounts);

    candidateIdx = ...
        find(tradeCounts==maxTrades);

    if maxTrades == 0

        error(['Ridge model generated ZERO validation trades ' ...
               'for the entire threshold grid. Increase the ' ...
               'threshold resolution or reconsider the model.']);

    end

    candidateReturns = ...
        thresholdResults(candidateIdx,2);

    [~,localBest] = max(candidateReturns);

    selectedIdx = candidateIdx(localBest);

    bestThreshold = ...
        thresholdResults(selectedIdx,1);

    bestValReturn = ...
        thresholdResults(selectedIdx,2);

    bestValSharpe = ...
        thresholdResults(selectedIdx,3);

end

fprintf('\n');
fprintf('Selected threshold     : %.4f%%\n', ...
    bestThreshold*100);

fprintf('Validation Sharpe      : %.4f\n', ...
    bestValSharpe);

fprintf('Validation return      : %.2f%%\n', ...
    bestValReturn*100);

fprintf('Validation trades      : %d\n', ...
    thresholdResults( ...
        find(thresholdResults(:,1)==bestThreshold,1),7));

%% ============================================================
% REFIT ON TRAIN + VALIDATION
%% ============================================================

fprintf('\n============================================================\n');
fprintf(' FINAL MODEL REFIT\n');
fprintf('============================================================\n\n');

XDevelopment = [XTrain;XVal];
yDevelopment = [yTrain;yVal];

muDevelopment = ...
    mean(XDevelopment,1);

sigmaDevelopment = ...
    std(XDevelopment,0,1);

sigmaDevelopment(sigmaDevelopment<eps) = 1;

XDevelopmentZ = ...
    (XDevelopment-muDevelopment)./sigmaDevelopment;

XTestFinalZ = ...
    (XTest-muDevelopment)./sigmaDevelopment;

finalModel = fitrlinear( ...
    XDevelopmentZ, ...
    yDevelopment, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda',bestLambda, ...
    'Solver','lbfgs', ...
    'FitBias',true);

%% ============================================================
% FINAL TEST PREDICTION
%% ============================================================

predTest = predict(finalModel,XTestFinalZ);

%% ============================================================
% TEST METRICS
%% ============================================================

testRMSE = sqrt(mean((predTest-yTest).^2));

testMAE = mean(abs(predTest-yTest));

testMSE = mean((predTest-yTest).^2);

SSres = sum((yTest-predTest).^2);
SStot = sum((yTest-mean(yTest)).^2);

if SStot > 0
    testR2 = 1-SSres/SStot;
else
    testR2 = NaN;
end

testDirectional = ...
    mean(sign(predTest)==sign(yTest))*100;

if std(predTest)>0 && std(yTest)>0

    predictionCorrelation = ...
        corr(predTest,yTest);

else

    predictionCorrelation = NaN;

end

%% ============================================================
% TEST SIGNALS
%% ============================================================

testPosition = zeros(size(predTest));

testPosition(predTest>bestThreshold) = 1;

testPosition(predTest<-bestThreshold) = -1;

numBuy = sum(testPosition==1);
numHold = sum(testPosition==0);
numSell = sum(testPosition==-1);

%% ============================================================
% TEST BACKTEST
%% ============================================================

[testStats,equityCurve,drawdownCurve] = ...
    runBacktest( ...
        yTest, ...
        testPosition, ...
        initialCapital, ...
        feeRate);

%% ============================================================
% BUY & HOLD
%% ============================================================

buyHoldPosition = ones(size(yTest));

[bhStats,bhEquity,bhDrawdown] = ...
    runBacktest( ...
        yTest, ...
        buyHoldPosition, ...
        initialCapital, ...
        feeRate);

%% ============================================================
% DEVELOPMENT OUTPUT
%% ============================================================

fprintf('\n============================================================\n');
fprintf(' DEVELOPMENT MODEL - VALIDATION\n');
fprintf('============================================================\n\n');

fprintf('Number of factors     : %d\n', ...
    numel(selectedFactors));

fprintf('Selected lambda       : %.6g\n', ...
    bestLambda);

fprintf('Train RMSE            : %.6f\n', ...
    trainRMSE);

fprintf('Train MAE             : %.6f\n', ...
    trainMAE);

fprintf('Validation RMSE       : %.6f\n', ...
    valRMSE);

fprintf('Validation MAE        : %.6f\n', ...
    valMAE);

fprintf('Validation R2         : %.6f\n', ...
    valR2);

fprintf('Directional accuracy  : %.2f%%\n', ...
    valDirectional);

fprintf('Selected threshold    : %.4f%%\n', ...
    bestThreshold*100);

fprintf('Validation Sharpe     : %.4f\n', ...
    bestValSharpe);

fprintf('Validation return     : %.2f%%\n', ...
    bestValReturn*100);

%% ============================================================
% TEST OUTPUT
%% ============================================================

fprintf('\n============================================================\n');
fprintf(' FINAL UNTOUCHED TEST\n');
fprintf('============================================================\n\n');

fprintf('Observations          : %d\n',numel(yTest));

fprintf('Start                 : %s\n', ...
    datestr(datesTest(1)));

fprintf('End                   : %s\n', ...
    datestr(datesTest(end)));

fprintf('RMSE                  : %.6f\n',testRMSE);

fprintf('MAE                   : %.6f\n',testMAE);

fprintf('MSE                   : %.6f\n',testMSE);

fprintf('R2                    : %.6f\n',testR2);

fprintf('Directional accuracy  : %.2f%%\n', ...
    testDirectional);

fprintf('Prediction correlation: %.4f\n', ...
    predictionCorrelation);

%% ============================================================
% STRATEGY OUTPUT
%% ============================================================

fprintf('\n============================================================\n');
fprintf(' FINAL TEST BACKTEST\n');
fprintf('============================================================\n\n');

fprintf('Initial capital       : $%.2f\n', ...
    initialCapital);

fprintf('Final capital         : $%.2f\n', ...
    testStats.finalCapital);

fprintf('Strategy return       : %.2f%%\n', ...
    testStats.return*100);

fprintf('Sharpe ratio          : %.4f\n', ...
    testStats.sharpe);

fprintf('Sortino ratio         : %.4f\n', ...
    testStats.sortino);

fprintf('Maximum drawdown      : %.2f%%\n', ...
    testStats.maxDrawdown*100);

fprintf('Win rate              : %.2f%%\n', ...
    testStats.winRate*100);

fprintf('Profit factor         : %.4f\n', ...
    testStats.profitFactor);

fprintf('Number of trades      : %d\n', ...
    testStats.trades);

fprintf('\nFinal test signals:\n');

fprintf('BUY                   = %d\n',numBuy);
fprintf('HOLD                  = %d\n',numHold);
fprintf('SELL                  = %d\n',numSell);

%% ============================================================
% BUY AND HOLD
%% ============================================================

fprintf('\n============================================================\n');
fprintf(' BUY & HOLD BENCHMARK\n');
fprintf('============================================================\n\n');

fprintf('Final capital         : $%.2f\n', ...
    bhStats.finalCapital);

fprintf('Return                : %.2f%%\n', ...
    bhStats.return*100);

fprintf('Sharpe ratio          : %.4f\n', ...
    bhStats.sharpe);

fprintf('Maximum drawdown      : %.2f%%\n', ...
    bhStats.maxDrawdown*100);

%% ============================================================
% COMPARISON
%% ============================================================

returnDifference = ...
    testStats.return-bhStats.return;

sharpeDifference = ...
    testStats.sharpe-bhStats.sharpe;

drawdownDifference = ...
    testStats.maxDrawdown-bhStats.maxDrawdown;

fprintf('\n============================================================\n');
fprintf(' RIDGE FACTOR vs BUY & HOLD\n');
fprintf('============================================================\n\n');

fprintf('Return difference     : %.2f percentage points\n', ...
    returnDifference*100);

fprintf('Sharpe difference     : %.4f\n', ...
    sharpeDifference);

fprintf('Drawdown difference   : %.2f percentage points\n', ...
    drawdownDifference*100);

%% ============================================================
% RIDGE COEFFICIENTS
%% ============================================================

try

    beta = finalModel.Beta(:);

catch

    beta = [];

end

if ~isempty(beta)

    coefficientTable = table( ...
        string(selectedFactors(:)), ...
        beta, ...
        abs(beta), ...
        'VariableNames', ...
        {'Factor','Coefficient','AbsoluteCoefficient'});

    coefficientTable = sortrows( ...
        coefficientTable, ...
        'AbsoluteCoefficient','descend');

else

    coefficientTable = table();

end

%% ============================================================
% PREDICTION TABLE
%
% All columns explicitly made column vectors.
%% ============================================================

testDates = datesTest(:);
actualReturns = yTest(:);
predictedReturns = predTest(:);
positions = testPosition(:);
equity = equityCurve(:);
drawdown = drawdownCurve(:);

predictionTable = table( ...
    testDates, ...
    actualReturns, ...
    predictedReturns, ...
    positions, ...
    equity, ...
    drawdown, ...
    'VariableNames', ...
    {'Date', ...
     'ActualReturn', ...
     'PredictedReturn', ...
     'Position', ...
     'StrategyEquity', ...
     'Drawdown'});

%% ============================================================
% SUMMARY TABLE
%% ============================================================

summaryMetric = { ...
    'Model'; ...
    'NumberOfFactors'; ...
    'SelectedLambda'; ...
    'SelectedThreshold'; ...
    'MinimumValidationTrades'; ...
    'TrainRMSE'; ...
    'ValidationRMSE'; ...
    'ValidationMAE'; ...
    'ValidationR2'; ...
    'ValidationDirectionalAccuracy'; ...
    'ValidationReturn'; ...
    'ValidationSharpe'; ...
    'TestRMSE'; ...
    'TestMAE'; ...
    'TestMSE'; ...
    'TestR2'; ...
    'TestDirectionalAccuracy'; ...
    'PredictionCorrelation'; ...
    'StrategyReturn'; ...
    'StrategySharpe'; ...
    'StrategySortino'; ...
    'StrategyMaxDrawdown'; ...
    'StrategyWinRate'; ...
    'StrategyProfitFactor'; ...
    'StrategyTrades'; ...
    'StrategyFinalCapital'; ...
    'BuyHoldReturn'; ...
    'BuyHoldSharpe'; ...
    'BuyHoldMaxDrawdown'; ...
    'BuyHoldFinalCapital'};

summaryValue = { ...
    'Multivariate Ridge Factor Model'; ...
    numel(selectedFactors); ...
    bestLambda; ...
    bestThreshold; ...
    minValidationTrades; ...
    trainRMSE; ...
    valRMSE; ...
    valMAE; ...
    valR2; ...
    valDirectional/100; ...
    bestValReturn; ...
    bestValSharpe; ...
    testRMSE; ...
    testMAE; ...
    testMSE; ...
    testR2; ...
    testDirectional/100; ...
    predictionCorrelation; ...
    testStats.return; ...
    testStats.sharpe; ...
    testStats.sortino; ...
    testStats.maxDrawdown; ...
    testStats.winRate; ...
    testStats.profitFactor; ...
    testStats.trades; ...
    testStats.finalCapital; ...
    bhStats.return; ...
    bhStats.sharpe; ...
    bhStats.maxDrawdown; ...
    bhStats.finalCapital};

summaryTable = table( ...
    summaryMetric, ...
    summaryValue, ...
    'VariableNames',{'Metric','Value'});

%% ============================================================
% LAMBDA TABLE
%% ============================================================

lambdaTable = array2table( ...
    lambdaResults, ...
    'VariableNames', ...
    {'Lambda', ...
     'ValidationRMSE', ...
     'ValidationMAE', ...
     'ValidationR2', ...
     'ValidationDirectionalAccuracy'});

%% ============================================================
% THRESHOLD TABLE
%% ============================================================

thresholdTable = array2table( ...
    thresholdResults, ...
    'VariableNames', ...
    {'Threshold', ...
     'ValidationReturn', ...
     'ValidationSharpe', ...
     'ValidationSortino', ...
     'ValidationMaxDrawdown', ...
     'ValidationWinRate', ...
     'ValidationTrades'});

%% ============================================================
% FACTOR LIST
%% ============================================================

factorTable = table( ...
    string(selectedFactors(:)), ...
    'VariableNames',{'Factor'});

%% ============================================================
% SAVE CSV FILES
%% ============================================================

predictionFile = fullfile( ...
    tablesDir, ...
    'STEP7_RIDGE_FACTOR_PREDICTIONS.csv');

summaryFile = fullfile( ...
    tablesDir, ...
    'STEP7_RIDGE_FACTOR_SUMMARY.csv');

coefficientFile = fullfile( ...
    tablesDir, ...
    'STEP7_RIDGE_FACTOR_COEFFICIENTS.csv');

lambdaFile = fullfile( ...
    tablesDir, ...
    'STEP7_RIDGE_LAMBDA_SELECTION.csv');

thresholdFile = fullfile( ...
    tablesDir, ...
    'STEP7_RIDGE_THRESHOLD_SELECTION.csv');

factorFile = fullfile( ...
    tablesDir, ...
    'STEP7_RIDGE_FACTOR_LIST.csv');

writetable(predictionTable,predictionFile);

writetable(summaryTable,summaryFile);

writetable(lambdaTable,lambdaFile);

writetable(thresholdTable,thresholdFile);

writetable(factorTable,factorFile);

if ~isempty(coefficientTable)

    writetable( ...
        coefficientTable, ...
        coefficientFile);

end

%% ============================================================
% SAVE MODEL
%% ============================================================

modelFile = fullfile( ...
    modelsDir, ...
    'STEP7_MULTIVARIATE_RIDGE_FACTOR_MODEL.mat');

save(modelFile, ...
    'finalModel', ...
    'developmentModel', ...
    'selectedFactors', ...
    'bestLambda', ...
    'bestThreshold', ...
    'minValidationTrades', ...
    'muTrain', ...
    'sigmaTrain', ...
    'muDevelopment', ...
    'sigmaDevelopment', ...
    'lambdaGrid', ...
    'thresholdGrid', ...
    'lambdaResults', ...
    'thresholdResults');

%% ============================================================
% FIGURE 1: EQUITY CURVE
%% ============================================================

figure('Color','w');

plot(testDates,equityCurve, ...
    'LineWidth',1.5);

hold on;

plot(testDates,bhEquity, ...
    'LineWidth',1.5);

grid on;

xlabel('Date');
ylabel('Portfolio Value ($)');

title('Step 7: Multivariate Ridge Factor Model vs Buy & Hold');

legend( ...
    'Ridge Factor Strategy', ...
    'Buy & Hold', ...
    'Location','best');

saveas(gcf, ...
    fullfile(figuresDir, ...
    'STEP7_RIDGE_EQUITY.png'));

%% ============================================================
% FIGURE 2: ACTUAL VS PREDICTED
%% ============================================================

figure('Color','w');

plot(testDates,yTest, ...
    'LineWidth',1.0);

hold on;

plot(testDates,predTest, ...
    'LineWidth',1.0);

grid on;

xlabel('Date');
ylabel('1-Day Return');

title('Step 7: Actual vs Predicted Returns');

legend( ...
    'Actual Return', ...
    'Predicted Return', ...
    'Location','best');

saveas(gcf, ...
    fullfile(figuresDir, ...
    'STEP7_RIDGE_ACTUAL_VS_PREDICTED.png'));

%% ============================================================
% FIGURE 3: DRAWDOWN
%% ============================================================

figure('Color','w');

plot(testDates,drawdownCurve*100, ...
    'LineWidth',1.5);

grid on;

xlabel('Date');
ylabel('Drawdown (%)');

title('Step 7: Ridge Factor Strategy Drawdown');

saveas(gcf, ...
    fullfile(figuresDir, ...
    'STEP7_RIDGE_DRAWDOWN.png'));

%% ============================================================
% FIGURE 4: FACTOR IMPORTANCE
%% ============================================================

if ~isempty(coefficientTable)

    figure('Color','w');

    barh(coefficientTable.AbsoluteCoefficient);

    set(gca, ...
        'YTick',1:height(coefficientTable), ...
        'YTickLabel',coefficientTable.Factor);

    xlabel('|Standardized Ridge Coefficient|');
    ylabel('Factor');

    title('Step 7: Ridge Factor Importance');

    grid on;

    saveas(gcf, ...
        fullfile(figuresDir, ...
        'STEP7_RIDGE_FACTOR_IMPORTANCE.png'));

end

%% ============================================================
% FIGURE 5: LAMBDA SELECTION
%% ============================================================

figure('Color','w');

semilogx( ...
    lambdaResults(:,1), ...
    lambdaResults(:,2), ...
    'LineWidth',1.5);

grid on;

xlabel('Ridge Lambda');
ylabel('Validation RMSE');

title('Step 7: Ridge Lambda Selection');

saveas(gcf, ...
    fullfile(figuresDir, ...
    'STEP7_RIDGE_LAMBDA_SELECTION.png'));

%% ============================================================
% FIGURE 6: THRESHOLD SELECTION
%% ============================================================

validThreshold = ...
    thresholdResults(:,7)>=minValidationTrades;

figure('Color','w');

if any(validThreshold)

    plot( ...
        thresholdResults(validThreshold,1)*100, ...
        thresholdResults(validThreshold,3), ...
        'LineWidth',1.5);

    grid on;

    xlabel('Threshold (%)');
    ylabel('Validation Sharpe');

else

    text(0.5,0.5, ...
        'No threshold met minimum trade requirement', ...
        'HorizontalAlignment','center');

    axis off;

end

title('Step 7: Trading Threshold Selection');

saveas(gcf, ...
    fullfile(figuresDir, ...
    'STEP7_RIDGE_THRESHOLD_SELECTION.png'));

%% ============================================================
% FINAL OUTPUT
%% ============================================================

fprintf('\n============================================================\n');
fprintf(' STEP 7 COMPLETED SUCCESSFULLY\n');
fprintf('============================================================\n\n');

fprintf('Factors used         : %d\n', ...
    numel(selectedFactors));

fprintf('Selected lambda      : %.6g\n', ...
    bestLambda);

fprintf('Selected threshold   : %.4f%%\n', ...
    bestThreshold*100);

fprintf('Test start           : %s\n', ...
    datestr(testDates(1)));

fprintf('Test end             : %s\n', ...
    datestr(testDates(end)));

fprintf('\nFiles saved:\n');

fprintf('1. %s\n',predictionFile);
fprintf('2. %s\n',summaryFile);
fprintf('3. %s\n',coefficientFile);
fprintf('4. %s\n',lambdaFile);
fprintf('5. %s\n',thresholdFile);
fprintf('6. %s\n',factorFile);
fprintf('7. %s\n',modelFile);

fprintf('\n============================================================\n');
fprintf(' IMPORTANT\n');
fprintf('============================================================\n\n');

fprintf(['The test period is fixed to the same period used by ' ...
         'the rigorous Step 5 AR(5) baseline.\n\n']);

fprintf(['Lambda and threshold selection use validation data only.\n\n']);

fprintf(['The final test data are not used during model selection.\n\n']);

fprintf('============================================================\n');


%% ============================================================
% LOCAL FUNCTION: NUMERIC CONVERSION
%% ============================================================

function x = convertToNumeric(value)

    if isnumeric(value)

        x = double(value);

    elseif islogical(value)

        x = double(value);

    elseif isstring(value) || ischar(value)

        x = str2double(string(value));

    elseif iscell(value)

        x = str2double(string(value));

    else

        error('Unsupported variable type.');

    end

    x = x(:);

end


%% ============================================================
% LOCAL FUNCTION: BACKTEST
%% ============================================================

function [stats,equityCurve,drawdownCurve] = ...
    runBacktest(actualReturns,positions,initialCapital,feeRate)

    actualReturns = actualReturns(:);
    positions = positions(:);

    n = numel(actualReturns);

    if numel(positions) ~= n
        error('Returns and positions must have equal length.');
    end

    %% Previous position

    previousPosition = zeros(n,1);

    if n > 1

        previousPosition(2:end) = ...
            positions(1:end-1);

    end

    %% Turnover

    turnover = ...
        abs(positions-previousPosition);

    %% Transaction costs

    transactionCost = ...
        feeRate.*turnover;

    %% Strategy return

    strategyReturn = ...
        positions.*actualReturns ...
        - transactionCost;

    %% Equity curve

    equityCurve = ...
        initialCapital.*cumprod(1+strategyReturn);

    %% Running maximum

    runningMaximum = ...
        cummax(equityCurve);

    %% Drawdown

    drawdownCurve = ...
        equityCurve./runningMaximum-1;

    %% Final capital

    finalCapital = equityCurve(end);

    totalReturn = ...
        finalCapital/initialCapital-1;

    %% Sharpe
    %
    % Same sqrt(365) convention as rigorous Step 5.
    %% 

    dailyMean = mean(strategyReturn);
    dailyStd = std(strategyReturn);

    if dailyStd > 0

        sharpe = ...
            sqrt(365)*dailyMean/dailyStd;

    else

        sharpe = 0;

    end

    %% Sortino

    downsideReturns = ...
        strategyReturn(strategyReturn<0);

    if ~isempty(downsideReturns)

        downsideDeviation = ...
            sqrt(mean(downsideReturns.^2));

        if downsideDeviation > 0

            sortino = ...
                sqrt(365)*dailyMean/downsideDeviation;

        else

            sortino = 0;

        end

    else

        sortino = 0;

    end

    %% Maximum drawdown

    maxDrawdown = ...
        min(drawdownCurve);

    %% Win rate

    activeReturns = ...
        strategyReturn(positions~=0);

    if isempty(activeReturns)

        winRate = 0;

    else

        winRate = ...
            mean(activeReturns>0);

    end

    %% Profit factor

    grossProfit = ...
        sum(strategyReturn(strategyReturn>0));

    grossLoss = ...
        -sum(strategyReturn(strategyReturn<0));

    if grossLoss > 0

        profitFactor = ...
            grossProfit/grossLoss;

    elseif grossProfit > 0

        profitFactor = Inf;

    else

        profitFactor = 0;

    end

    %% Number of trades

    trades = ...
        sum(turnover>0);

    %% Output

    stats.initialCapital = initialCapital;
    stats.finalCapital = finalCapital;
    stats.return = totalReturn;
    stats.sharpe = sharpe;
    stats.sortino = sortino;
    stats.maxDrawdown = maxDrawdown;
    stats.winRate = winRate;
    stats.profitFactor = profitFactor;
    stats.trades = trades;

end