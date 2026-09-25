function lib = factorTradingLib()
%FACTORTRADINGLIB Shared functions for the Crypto Sentiment Factor Model.
%
% Usage:
%   lib = factorTradingLib();
%   cfg = lib.config();
%   merged = lib.loadAndMergeData(btcFile, sentimentFile);
%   [X, factorNames] = lib.engineerFactors(merged);
%
% The file uses one public entry-point function that returns a structure
% containing handles to all helper functions.

lib.config                 = @config;
lib.findColumnIndex        = @findColumnIndex;
lib.loadAndMergeData       = @loadAndMergeData;
lib.engineerFactors        = @engineerFactors;
lib.createReturnTarget     = @createReturnTarget;
lib.trainValTestSplit      = @trainValTestSplit;
lib.fitFactorModel         = @fitFactorModel;
lib.predictReturns         = @predictReturns;
lib.returnsToSignal        = @returnsToSignal;
lib.regressionReport       = @regressionReport;
lib.runBacktest            = @runBacktest;
lib.plotFactorSignificance = @plotFactorSignificance;
lib.plotActualVsPredicted  = @plotActualVsPredicted;
lib.plotEquityCurve        = @plotEquityCurve;
lib.plotDrawdown           = @plotDrawdown;
lib.plotSignalChart        = @plotSignalChart;

end


%% ============================================================
% CONFIGURATION
% ============================================================
function cfg = config()
%CONFIG Default configuration for the factor-model pipeline.

cfg = struct();

% --- Target ---
cfg.predictionHorizon = 1;          % Predict next-day return

% --- Factor model ---
cfg.stepwiseAlphaEnter = 0.05;
cfg.stepwiseAlphaLeave = 0.10;

% --- Trading thresholds ---
% BUY  when predicted return > buyThreshold
% SELL when predicted return < -sellThreshold
% HOLD otherwise
cfg.buyThreshold  = 0.0;
cfg.sellThreshold = 0.0;

% --- Chronological data split ---
cfg.trainRatio      = 0.60;
cfg.validationRatio = 0.20;
% Remaining 20% is the untouched test set.

% --- Backtest ---
cfg.initialCapital = 100;

end


%% ============================================================
% COLUMN SEARCH
% ============================================================
function idx = findColumnIndex(varNames, exactCandidates, substringCandidates)
%FINDCOLUMNINDEX Find a table column by exact or substring matching.

idx = [];

% Exact match
for i = 1:numel(exactCandidates)
    match = find(strcmpi(string(varNames), string(exactCandidates{i})), 1);

    if ~isempty(match)
        idx = match;
        return;
    end
end

% Substring fallback
if nargin >= 3 && ~isempty(substringCandidates)

    if ischar(substringCandidates) || isstring(substringCandidates)
        substringCandidates = cellstr(substringCandidates);
    end

    lowerNames = lower(string(varNames));

    for i = 1:numel(lowerNames)
        for j = 1:numel(substringCandidates)

            candidate = lower(string(substringCandidates{j}));

            if contains(lowerNames(i), candidate)
                idx = i;
                return;
            end

        end
    end
end

end


%% ============================================================
% DATA LOADING / ALIGNMENT
% ============================================================
function merged = loadAndMergeData(btcFile, sentimentFile)
%LOADANDMERGEDATA Load and merge BTC price and sentiment data.
%
% Output columns:
%   Date
%   Open
%   High
%   Low
%   Close
%   Volume
%   NumberOfTrades
%   Sentiment
%   SampleSize

if ~isfile(btcFile)
    error('loadAndMergeData:missingBTCFile', ...
        'BTC file does not exist: %s', btcFile);
end

if ~isfile(sentimentFile)
    error('loadAndMergeData:missingSentimentFile', ...
        'Sentiment file does not exist: %s', sentimentFile);
end


%% --- BTC PRICE DATA ---

btc = readtable(btcFile);
btcVarNames = btc.Properties.VariableNames;

% Find Open time
openTimeIdx = findColumnIndex( ...
    btcVarNames, ...
    {'Open_time','OpenTime','Open time'}, ...
    {'open time','opentime'});

if isempty(openTimeIdx)
    error('loadAndMergeData:noOpenTime', ...
        'Could not find the BTC Open time column in %s', btcFile);
end

% Find OHLCV columns
openIdx = findColumnIndex( ...
    btcVarNames, ...
    {'Open'}, ...
    {'open'});

highIdx = findColumnIndex( ...
    btcVarNames, ...
    {'High'}, ...
    {'high'});

lowIdx = findColumnIndex( ...
    btcVarNames, ...
    {'Low'}, ...
    {'low'});

closeIdx = findColumnIndex( ...
    btcVarNames, ...
    {'Close'}, ...
    {'close'});

volIdx = findColumnIndex( ...
    btcVarNames, ...
    {'Volume'}, ...
    {'volume'});

tradesIdx = findColumnIndex( ...
    btcVarNames, ...
    {'Number_of_trades','NumberOfTrades','Number of trades'}, ...
    {'number of trades','trades'});

if isempty(openIdx) || isempty(highIdx) || isempty(lowIdx) || ...
        isempty(closeIdx) || isempty(volIdx)

    error('loadAndMergeData:missingPriceColumns', ...
        'One or more required BTC OHLCV columns are missing.');
end


%% --- Parse BTC dates ---

rawDates = btc.(btcVarNames{openTimeIdx});

if isdatetime(rawDates)

    btcDates = dateshift(rawDates, 'start', 'day');

else

    rawDates = string(rawDates);

    % Remove UTC suffix
    rawDates = erase(rawDates, " UTC");

    % Parse Binance-style timestamps
    try
        btcDates = datetime( ...
            rawDates, ...
            'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSSSS');
    catch
        try
            btcDates = datetime(rawDates);
        catch ME
            error('loadAndMergeData:invalidBTCDate', ...
                'Unable to parse BTC dates. MATLAB error: %s', ME.message);
        end
    end

    btcDates = dateshift(btcDates, 'start', 'day');

end


%% --- Construct BTC table ---

priceTable = table();

priceTable.Date = btcDates;

priceTable.Open = double(btc.(btcVarNames{openIdx}));
priceTable.High = double(btc.(btcVarNames{highIdx}));
priceTable.Low = double(btc.(btcVarNames{lowIdx}));
priceTable.Close = double(btc.(btcVarNames{closeIdx}));
priceTable.Volume = double(btc.(btcVarNames{volIdx}));

if ~isempty(tradesIdx)
    priceTable.NumberOfTrades = ...
        double(btc.(btcVarNames{tradesIdx}));
else
    priceTable.NumberOfTrades = ...
        NaN(height(priceTable), 1);
end


%% --- Remove invalid BTC rows ---

validPrice = ...
    ~isnat(priceTable.Date) & ...
    isfinite(priceTable.Open) & ...
    isfinite(priceTable.High) & ...
    isfinite(priceTable.Low) & ...
    isfinite(priceTable.Close) & ...
    isfinite(priceTable.Volume) & ...
    priceTable.Close > 0 & ...
    priceTable.Volume >= 0;

priceTable = priceTable(validPrice, :);


%% --- Remove duplicate dates ---

[~, uniqueIdx] = unique(priceTable.Date, 'last');

priceTable = priceTable(sort(uniqueIdx), :);

priceTable = sortrows(priceTable, 'Date');


%% ============================================================
% SENTIMENT DATA
% ============================================================

sent = readtable(sentimentFile);
sentVarNames = sent.Properties.VariableNames;

sentDateIdx = findColumnIndex( ...
    sentVarNames, ...
    {'time','Date','date'}, ...
    {'date','time'});

sentScoreIdx = findColumnIndex( ...
    sentVarNames, ...
    {'sentiment','Sentiment'}, ...
    {'sentiment'});

sentSampleIdx = findColumnIndex( ...
    sentVarNames, ...
    {'sample_size','SampleSize','sample size'}, ...
    {'sample'});

if isempty(sentDateIdx)
    error('loadAndMergeData:noSentimentDate', ...
        'Could not find sentiment date column.');
end

if isempty(sentScoreIdx)
    error('loadAndMergeData:noSentimentScore', ...
        'Could not find sentiment score column.');
end


%% --- Parse sentiment dates ---

sentDates = sent.(sentVarNames{sentDateIdx});

if ~isdatetime(sentDates)

    try
        sentDates = datetime(sentDates);
    catch ME
        error('loadAndMergeData:invalidSentimentDate', ...
            'Unable to parse sentiment dates. MATLAB error: %s', ...
            ME.message);
    end

end

sentDates = dateshift(sentDates, 'start', 'day');


%% --- Construct sentiment table ---

sentTable = table();

sentTable.Date = sentDates;

sentTable.Sentiment = ...
    double(sent.(sentVarNames{sentScoreIdx}));

if ~isempty(sentSampleIdx)

    sentTable.SampleSize = ...
        double(sent.(sentVarNames{sentSampleIdx}));

else

    sentTable.SampleSize = ...
        NaN(height(sentTable), 1);

end


%% --- Remove invalid sentiment rows ---

validSentiment = ...
    ~isnat(sentTable.Date) & ...
    isfinite(sentTable.Sentiment);

sentTable = sentTable(validSentiment, :);


%% --- Remove duplicate sentiment dates ---

[~, uniqueIdx] = unique(sentTable.Date, 'last');

sentTable = sentTable(sort(uniqueIdx), :);

sentTable = sortrows(sentTable, 'Date');


%% ============================================================
% MERGE
% ============================================================

merged = innerjoin( ...
    priceTable, ...
    sentTable, ...
    'Keys', 'Date');

merged = sortrows(merged, 'Date');


if isempty(merged)

    error('loadAndMergeData:noOverlap', ...
        ['No overlapping dates were found between the BTC price ' ...
         'and sentiment datasets.']);

end

end


%% ============================================================
% WILDER SMOOTHING
% ============================================================
function out = wilderSmooth(x, period)
%WILDERSMOOTH Wilder smoothing used for RSI.

x = x(:);
n = length(x);

out = NaN(n, 1);

if n < period
    return;
end

% Find the first valid starting window
firstWindow = x(1:period);

if all(isnan(firstWindow))
    return;
end

out(period) = mean(firstWindow, 'omitnan');

for t = period + 1:n

    if isnan(x(t))

        out(t) = out(t-1);

    else

        out(t) = ...
            ((period - 1) * out(t-1) + x(t)) / period;

    end

end

end


%% ============================================================
% EXPONENTIAL MOVING AVERAGE
% ============================================================
function out = ema(x, period)
%EMA Exponential moving average.

x = x(:);
n = length(x);

out = NaN(n, 1);

if n < period
    return;
end

alpha = 2 / (period + 1);

initialWindow = x(1:period);

if all(isnan(initialWindow))
    return;
end

out(period) = mean(initialWindow, 'omitnan');

for t = period + 1:n

    if isnan(x(t))

        out(t) = out(t-1);

    else

        out(t) = ...
            alpha * x(t) + ...
            (1 - alpha) * out(t-1);

    end

end

end


%% ============================================================
% FACTOR ENGINEERING
% ============================================================
function [X, factorNames] = engineerFactors(merged)
%ENGINEERFACTORS Create price, technical, volume and sentiment factors.
%
% Factors:
%
% 1. Return1D
% 2. Return3D
% 3. Return7D
% 4. PriceVsSMA14
% 5. Volatility7
% 6. Volatility14
% 7. RSI14
% 8. MACD
% 9. VolumeChange
% 10. RelativeVolume
% 11. Sentiment
% 12. SentimentChange
% 13. SentimentMA7
% 14. SentimentReturnInteraction

requiredColumns = ...
    {'Close','Volume','Sentiment'};

for i = 1:numel(requiredColumns)

    if ~ismember(requiredColumns{i}, merged.Properties.VariableNames)

        error('engineerFactors:missingColumn', ...
            'Required column missing: %s', requiredColumns{i});

    end

end


price = double(merged.Close(:));
volume = double(merged.Volume(:));
sentiment = double(merged.Sentiment(:));

N = length(price);

if length(volume) ~= N || length(sentiment) ~= N

    error('engineerFactors:lengthMismatch', ...
        'Price, volume and sentiment vectors must have equal length.');

end


%% --- Price returns ---

returns1 = NaN(N, 1);

valid = ...
    isfinite(price(2:end)) & ...
    isfinite(price(1:end-1)) & ...
    price(1:end-1) ~= 0;

returns1(2:end) = ...
    (price(2:end) - price(1:end-1)) ./ ...
    price(1:end-1);

returns1(~valid) = NaN;


returns3 = NaN(N, 1);

if N > 3

    valid = ...
        isfinite(price(4:end)) & ...
        isfinite(price(1:end-3)) & ...
        price(1:end-3) ~= 0;

    temp = ...
        (price(4:end) - price(1:end-3)) ./ ...
        price(1:end-3);

    temp(~valid) = NaN;

    returns3(4:end) = temp;

end


returns7 = NaN(N, 1);

if N > 7

    valid = ...
        isfinite(price(8:end)) & ...
        isfinite(price(1:end-7)) & ...
        price(1:end-7) ~= 0;

    temp = ...
        (price(8:end) - price(1:end-7)) ./ ...
        price(1:end-7);

    temp(~valid) = NaN;

    returns7(8:end) = temp;

end


%% --- Price relative to SMA ---

sma14 = movmean( ...
    price, ...
    [13 0], ...
    'omitnan');

priceVsSMA14 = ...
    price ./ sma14 - 1;


%% --- Historical volatility ---

volatility7 = movstd( ...
    returns1, ...
    [6 0], ...
    'omitnan');

volatility14 = movstd( ...
    returns1, ...
    [13 0], ...
    'omitnan');


%% ============================================================
% RSI 14
% ============================================================

delta = [NaN; diff(price)];

gain = max(delta, 0);
loss = max(-delta, 0);

% Ignore the artificial first NaN
gain(1) = NaN;
loss(1) = NaN;

avgGain = wilderSmooth(gain, 14);
avgLoss = wilderSmooth(loss, 14);

rsi14 = NaN(N, 1);

validRSI = ...
    isfinite(avgGain) & ...
    isfinite(avgLoss);

% Normal RSI calculation
rsi14(validRSI) = ...
    100 - 100 ./ ...
    (1 + avgGain(validRSI) ./ max(avgLoss(validRSI), eps));

% If average loss is exactly zero, RSI should be 100
zeroLoss = validRSI & avgLoss == 0 & avgGain > 0;
rsi14(zeroLoss) = 100;


%% ============================================================
% MACD
% ============================================================

ema12 = ema(price, 12);
ema26 = ema(price, 26);

macd = ema12 - ema26;


%% ============================================================
% VOLUME FACTORS
% ============================================================

volumeChange = NaN(N, 1);

valid = ...
    isfinite(volume(2:end)) & ...
    isfinite(volume(1:end-1)) & ...
    volume(1:end-1) ~= 0;

volumeChange(2:end) = ...
    (volume(2:end) - volume(1:end-1)) ./ ...
    volume(1:end-1);

volumeChange(2:end) = ...
    replaceInvalid(volumeChange(2:end), valid);


volumeMA7 = movmean( ...
    volume, ...
    [6 0], ...
    'omitnan');

relativeVolume = ...
    volume ./ volumeMA7;


%% ============================================================
% SENTIMENT FACTORS
% ============================================================

sentimentChange = [NaN; diff(sentiment)];

sentimentMA7 = movmean( ...
    sentiment, ...
    [6 0], ...
    'omitnan');

sentimentReturnInteraction = ...
    sentiment .* returns1;


%% ============================================================
% FINAL FEATURE MATRIX
% ============================================================

X = [ ...
    returns1, ...
    returns3, ...
    returns7, ...
    priceVsSMA14, ...
    volatility7, ...
    volatility14, ...
    rsi14, ...
    macd, ...
    volumeChange, ...
    relativeVolume, ...
    sentiment, ...
    sentimentChange, ...
    sentimentMA7, ...
    sentimentReturnInteraction];


factorNames = { ...
    'Return1D'
    'Return3D'
    'Return7D'
    'PriceVsSMA14'
    'Volatility7'
    'Volatility14'
    'RSI14'
    'MACD'
    'VolumeChange'
    'RelativeVolume'
    'Sentiment'
    'SentimentChange'
    'SentimentMA7'
    'SentimentReturnInteraction'};


%% --- Validation ---

assert( ...
    size(X, 2) == numel(factorNames), ...
    'engineerFactors:mismatch', ...
    'Factor matrix has %d columns but %d names were defined.', ...
    size(X,2), numel(factorNames));

assert( ...
    size(X,1) == N, ...
    'engineerFactors:rowMismatch', ...
    'Factor matrix has %d rows but %d input rows were given.', ...
    size(X,1), N);

end


%% ============================================================
% HELPER: REPLACE INVALID VALUES
% ============================================================
function out = replaceInvalid(x, validMask)
%REPLACEINVALID Set values to NaN where mask is false.

out = x;

out(~validMask) = NaN;

end


%% ============================================================
% RETURN TARGET
% ============================================================
function [y, validMask] = createReturnTarget(price, predictionHorizon)
%CREATERETURNTARGET Create forward return target.
%
% For horizon = 1:
%
%   y(t) = (Price(t+1) - Price(t)) / Price(t)
%
% The final predictionHorizon rows do not have future prices and
% therefore receive NaN.

price = double(price(:));

N = length(price);

if ~isscalar(predictionHorizon) || ...
        predictionHorizon < 1 || ...
        predictionHorizon ~= floor(predictionHorizon)

    error('createReturnTarget:invalidHorizon', ...
        'predictionHorizon must be a positive integer.');

end

if predictionHorizon >= N

    error('createReturnTarget:shortData', ...
        'Prediction horizon must be smaller than the number of observations.');

end


y = NaN(N, 1);

futurePrice = price(1 + predictionHorizon:N);
currentPrice = price(1:N - predictionHorizon);

valid = ...
    isfinite(currentPrice) & ...
    isfinite(futurePrice) & ...
    currentPrice ~= 0;

temp = NaN(size(currentPrice));

temp(valid) = ...
    (futurePrice(valid) - currentPrice(valid)) ./ ...
    currentPrice(valid);

y(1:N - predictionHorizon) = temp;

validMask = isfinite(y);

end


%% ============================================================
% CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
% ============================================================
function [trainIdx, validationIdx, testIdx] = ...
    trainValTestSplit(N, trainRatio, validationRatio)
%TRAINVALTESTSPLIT Chronological train/validation/test split.

if N < 10
    error('trainValTestSplit:tooFewSamples', ...
        'At least 10 samples are recommended for splitting.');
end

if trainRatio <= 0 || validationRatio < 0
    error('trainValTestSplit:invalidRatio', ...
        'Train and validation ratios must be non-negative.');
end

if trainRatio + validationRatio >= 1
    error('trainValTestSplit:invalidRatio', ...
        'Train ratio + validation ratio must be less than 1.');
end


trainEnd = floor(trainRatio * N);

validationEnd = ...
    floor((trainRatio + validationRatio) * N);


trainIdx = 1:trainEnd;

validationIdx = ...
    (trainEnd + 1):validationEnd;

testIdx = ...
    (validationEnd + 1):N;


if isempty(trainIdx) || ...
        isempty(validationIdx) || ...
        isempty(testIdx)

    error('trainValTestSplit:emptySplit', ...
        'One of the train/validation/test splits is empty.');

end

end


%% ============================================================
% FACTOR MODEL
% ============================================================
function modelInfo = fitFactorModel(XTrain, yTrain, factorNames, cfg)
%FITFACTORMODEL Fit linear regression using stepwise factor selection.

if isempty(XTrain) || isempty(yTrain)
    error('fitFactorModel:emptyData', ...
        'Training data cannot be empty.');
end

if size(XTrain,1) ~= length(yTrain)
    error('fitFactorModel:lengthMismatch', ...
        'XTrain and yTrain must have the same number of rows.');
end

if size(XTrain,2) ~= numel(factorNames)
    error('fitFactorModel:factorMismatch', ...
        'Number of factor names does not match XTrain columns.');
end


validRows = ...
    all(isfinite(XTrain), 2) & ...
    isfinite(yTrain);

XTrain = XTrain(validRows, :);
yTrain = yTrain(validRows);


if size(XTrain,1) < 50
    warning('fitFactorModel:smallTrainingSet', ...
        'Only %d valid training observations remain.', ...
        size(XTrain,1));
end


validNames = matlab.lang.makeValidName(factorNames);

% Ensure predictor names are unique
validNames = matlab.lang.makeUniqueStrings(validNames);


XTable = array2table( ...
    XTrain, ...
    'VariableNames', validNames);

XTable.Target = yTrain;


%% --- Stepwise regression ---

model = stepwiselm( ...
    XTable, ...
    'Target ~ 1', ...
    'Upper', 'linear', ...
    'PEnter', cfg.stepwiseAlphaEnter, ...
    'PRemove', cfg.stepwiseAlphaLeave, ...
    'Verbose', 0);


coefTable = model.Coefficients;

rowNames = coefTable.Properties.RowNames;

selectedNames = ...
    rowNames(~strcmp(rowNames, '(Intercept)'));


modelInfo = struct();

modelInfo.model = model;

modelInfo.coefficients = coefTable;

modelInfo.selectedNames = selectedNames;

modelInfo.rSquared = ...
    model.Rsquared.Ordinary;

modelInfo.adjRSquared = ...
    model.Rsquared.Adjusted;

modelInfo.factorNames = factorNames;

modelInfo.validNameMap = validNames;

modelInfo.trainingRows = size(XTrain,1);

end


%% ============================================================
% PREDICT RETURNS
% ============================================================
function predictedReturn = predictReturns(modelInfo, X)
%PREDICTRETURNS Generate predictions from fitted model.

if isempty(X)
    predictedReturn = zeros(0,1);
    return;
end

if size(X,2) ~= numel(modelInfo.validNameMap)

    error('predictReturns:factorMismatch', ...
        ['Prediction matrix contains %d columns, but the model ' ...
         'expects %d columns.'], ...
        size(X,2), numel(modelInfo.validNameMap));

end


XTable = array2table( ...
    X, ...
    'VariableNames', modelInfo.validNameMap);


predictedReturn = predict( ...
    modelInfo.model, ...
    XTable);

predictedReturn = predictedReturn(:);

end


%% ============================================================
% RETURN -> SIGNAL
% ============================================================
function signal = returnsToSignal( ...
    predictedReturn, buyThreshold, sellThreshold)
%RETURNSTOSIGNAL Convert predicted returns into BUY/HOLD/SELL.

if buyThreshold < 0 || sellThreshold < 0

    error('returnsToSignal:invalidThreshold', ...
        'Trading thresholds cannot be negative.');

end


predictedReturn = predictedReturn(:);

signal = repmat( ...
    "HOLD", ...
    size(predictedReturn));


signal(predictedReturn > buyThreshold) = ...
    "BUY";

signal(predictedReturn < -sellThreshold) = ...
    "SELL";

end


%% ============================================================
% REGRESSION METRICS
% ============================================================
function report = regressionReport(actual, predicted)
%REGRESSIONREPORT Calculate regression and directional metrics.

actual = actual(:);
predicted = predicted(:);

if length(actual) ~= length(predicted)

    error('regressionReport:lengthMismatch', ...
        'Actual and predicted vectors must have equal length.');

end


valid = ...
    isfinite(actual) & ...
    isfinite(predicted);

actual = actual(valid);
predicted = predicted(valid);


if isempty(actual)

    error('regressionReport:noValidData', ...
        'No valid observations are available.');

end


err = actual - predicted;

report = struct();


%% --- RMSE ---

report.rmse = ...
    sqrt(mean(err.^2));


%% --- MAE ---

report.mae = ...
    mean(abs(err));


%% --- R-squared ---

ssRes = sum(err.^2);

meanActual = mean(actual);

ssTot = ...
    sum((actual - meanActual).^2);

if ssTot > 0

    report.rSquared = ...
        1 - ssRes / ssTot;

else

    report.rSquared = 0;

end


%% --- Directional accuracy ---

actualSign = sign(actual);
predictedSign = sign(predicted);

report.directionalAccuracy = ...
    mean(actualSign == predictedSign);


end


%% ============================================================
% BACKTEST
% ============================================================
function result = runBacktest( ...
    testPrices, signals, initialCapital)
%RUNBACKTEST Long/flat/short daily strategy.
%
% BUY  -> +1
% SELL -> -1
% HOLD -> 0
%
% Signal at day t is applied to the return from t to t+1.

testPrices = double(testPrices(:));
signals = string(signals(:));

nDays = length(testPrices);

if length(signals) ~= nDays

    error('runBacktest:lengthMismatch', ...
        'Number of prices and signals must be equal.');

end

if nDays < 2

    error('runBacktest:tooFewDays', ...
        'At least two test prices are required.');

end

if initialCapital <= 0

    error('runBacktest:invalidCapital', ...
        'Initial capital must be greater than zero.');

end

if any(~isfinite(testPrices)) || ...
        any(testPrices <= 0)

    error('runBacktest:invalidPrices', ...
        'Test prices must be finite and greater than zero.');

end


capital = initialCapital;

equity = zeros(nDays, 1);

equity(1) = initialCapital;


position = 0;

positionChanges = 0;

winningDays = 0;
losingDays = 0;

dailyReturn = zeros(nDays - 1, 1);


%% --- Main backtest loop ---

for i = 1:nDays-1

    signal = signals(i);


    if signal == "BUY"

        newPosition = 1;

    elseif signal == "SELL"

        newPosition = -1;

    else

        newPosition = 0;

    end


    % Count position changes
    if newPosition ~= position
        positionChanges = positionChanges + 1;
    end

    position = newPosition;


    % Close-to-close market return
    marketReturn = ...
        (testPrices(i+1) - testPrices(i)) / ...
        testPrices(i);


    strategyReturn = ...
        position * marketReturn;


    dailyReturn(i) = strategyReturn;


    % Compound capital
    capital = ...
        capital * (1 + strategyReturn);


    equity(i+1) = capital;


    if strategyReturn > 0

        winningDays = winningDays + 1;

    elseif strategyReturn < 0

        losingDays = losingDays + 1;

    end

end


%% --- Overall returns ---

strategyReturnPercent = ...
    100 * (capital / initialCapital - 1);


buyHoldReturnPercent = ...
    100 * (testPrices(end) / testPrices(1) - 1);


%% --- Sharpe ratio ---

dailyStd = std(dailyReturn);

if isfinite(dailyStd) && dailyStd > 0

    sharpeRatio = ...
        mean(dailyReturn) / dailyStd * sqrt(365);

else

    sharpeRatio = 0;

end


%% --- Drawdown ---

runningMax = cummax(equity);

drawdown = ...
    (equity - runningMax) ./ runningMax;


maximumDrawdownPercent = ...
    100 * min(drawdown);


%% --- Win rate ---

totalDays = ...
    winningDays + losingDays;

if totalDays > 0

    winRatePercent = ...
        100 * winningDays / totalDays;

else

    winRatePercent = 0;

end


%% --- Output ---

result = struct();

result.equity = equity;

result.dailyReturn = dailyReturn;

result.drawdown = drawdown;

result.capital = capital;

result.tradeCount = positionChanges;

result.winningDays = winningDays;

result.losingDays = losingDays;

result.strategyReturnPercent = ...
    strategyReturnPercent;

result.buyHoldReturnPercent = ...
    buyHoldReturnPercent;

result.sharpeRatio = sharpeRatio;

result.maximumDrawdownPercent = ...
    maximumDrawdownPercent;

result.winRatePercent = ...
    winRatePercent;

end


%% ============================================================
% PLOT FACTOR SIGNIFICANCE
% ============================================================
function plotFactorSignificance( ...
    coefficients, figureTitle, savePath)
%PLOTFACTORSIGNIFICANCE Plot selected factor coefficients and p-values.

names = coefficients.Properties.RowNames;

keep = ...
    ~strcmp(names, '(Intercept)');

names = names(keep);

estimates = ...
    coefficients.Estimate(keep);

pValues = ...
    coefficients.pValue(keep);


figure('Name', figureTitle);

if isempty(names)

    text(0.5, 0.5, ...
        'No factors selected by stepwise regression.', ...
        'HorizontalAlignment', 'center');

    axis off;

else

    bar(estimates);

    set(gca, ...
        'XTick', 1:numel(names), ...
        'XTickLabel', names, ...
        'XTickLabelRotation', 45);

    xlabel('Factor');

    ylabel('Coefficient');

    title(figureTitle);

    grid on;


    for i = 1:numel(names)

        if estimates(i) >= 0
            verticalAlignment = 'bottom';
        else
            verticalAlignment = 'top';
        end

        text( ...
            i, ...
            estimates(i), ...
            sprintf('p=%.3f', pValues(i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', verticalAlignment);

    end

end


if nargin >= 3 && ~isempty(savePath)

    saveas(gcf, savePath);

    close(gcf);

end

end


%% ============================================================
% ACTUAL VS PREDICTED
% ============================================================
function plotActualVsPredicted( ...
    actual, predicted, figureTitle, savePath)

figure('Name', figureTitle);

plot(actual, 'LineWidth', 1.2);

hold on;

plot(predicted, 'LineWidth', 1.2);

legend( ...
    'Actual return', ...
    'Predicted return', ...
    'Location', 'best');

xlabel('Test Day');

ylabel('Return');

title(figureTitle);

grid on;

hold off;


if nargin >= 4 && ~isempty(savePath)

    saveas(gcf, savePath);

    close(gcf);

end

end


%% ============================================================
% EQUITY CURVE
% ============================================================
function plotEquityCurve(equity, figureTitle, savePath)

figure('Name', figureTitle);

plot(equity, 'LineWidth', 1.5);

xlabel('Test Day');

ylabel('Portfolio Value ($)');

title(figureTitle);

grid on;


if nargin >= 3 && ~isempty(savePath)

    saveas(gcf, savePath);

    close(gcf);

end

end


%% ============================================================
% DRAWDOWN
% ============================================================
function plotDrawdown(drawdown, figureTitle, savePath)

figure('Name', figureTitle);

plot( ...
    100 * drawdown, ...
    'LineWidth', 1.5);

xlabel('Test Day');

ylabel('Drawdown (%)');

title(figureTitle);

grid on;


if nargin >= 3 && ~isempty(savePath)

    saveas(gcf, savePath);

    close(gcf);

end

end


%% ============================================================
% SIGNAL CHART
% ============================================================
function plotSignalChart( ...
    prices, signals, figureTitle, savePath)

prices = prices(:);
signals = string(signals(:));

if length(prices) ~= length(signals)

    error('plotSignalChart:lengthMismatch', ...
        'Prices and signals must have equal length.');

end


figure('Name', figureTitle);

plot(prices, 'LineWidth', 1.2);

hold on;


buyIdx = find(signals == "BUY");

holdIdx = find(signals == "HOLD");

sellIdx = find(signals == "SELL");


if ~isempty(buyIdx)

    plot( ...
        buyIdx, ...
        prices(buyIdx), ...
        '^', ...
        'MarkerSize', 6);

end


if ~isempty(holdIdx)

    plot( ...
        holdIdx, ...
        prices(holdIdx), ...
        'o', ...
        'MarkerSize', 5);

end


if ~isempty(sellIdx)

    plot( ...
        sellIdx, ...
        prices(sellIdx), ...
        'v', ...
        'MarkerSize', 6);

end


xlabel('Test Day');

ylabel('BTC Price');

title(figureTitle);

legend( ...
    'BTC Price', ...
    'BUY', ...
    'HOLD', ...
    'SELL', ...
    'Location', 'best');

grid on;

hold off;


if nargin >= 4 && ~isempty(savePath)

    saveas(gcf, savePath);

    close(gcf);

end

end