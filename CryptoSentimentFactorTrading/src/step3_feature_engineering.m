function step3_feature_engineering()
%STEP3_FEATURE_ENGINEERING
% Create price, technical-analysis and sentiment features.
%
% Input:
%   data/processed/sentiment_classified_dataset.csv
%
% Output:
%   data/processed/factor_dataset.csv
%
% Features include:
%   - Daily returns
%   - Log returns
%   - Lagged returns
%   - Moving averages
%   - EMA
%   - RSI
%   - MACD
%   - Volatility
%   - Volume change
%   - Sentiment features
%   - Sentiment regime
%
% Important:
% Features are constructed using information available up to the
% corresponding date. The prediction target is the FUTURE return.

clc;
close all;

fprintf('\n============================================================\n');
fprintf(' [STEP 3] FEATURE ENGINEERING\n');
fprintf('============================================================\n\n');

%% -----------------------------------------------------------
% 1. PROJECT PATHS
% ------------------------------------------------------------

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

%% -----------------------------------------------------------
% 2. FILE PATHS
% ------------------------------------------------------------

inputFile = fullfile( ...
    projectDir, ...
    'data', ...
    'processed', ...
    'sentiment_classified_dataset.csv');

outputFile = fullfile( ...
    projectDir, ...
    'data', ...
    'processed', ...
    'factor_dataset.csv');

figureDir = fullfile( ...
    projectDir, ...
    'results', ...
    'figures');

if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

%% -----------------------------------------------------------
% 3. CHECK INPUT
% ------------------------------------------------------------

if ~isfile(inputFile)

    error('step3:missingInput', ...
        ['Sentiment-classified dataset was not found:\n%s\n\n' ...
         'Run Step 2 first.'], ...
        inputFile);

end

%% -----------------------------------------------------------
% 4. LOAD DATA
% ------------------------------------------------------------

fprintf('-> Loading sentiment-classified dataset...\n');
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

        error('step3:readError', ...
            'Unable to read input dataset.\n%s', ...
            ME.message);

    end

end

if isempty(data) || height(data) == 0

    error('step3:emptyData', ...
        'Input dataset contains no observations.');

end

fprintf('   Input observations: %d\n', ...
    height(data));

%% -----------------------------------------------------------
% 5. VALIDATE REQUIRED COLUMNS
% ------------------------------------------------------------

requiredColumns = { ...
    'Date', ...
    'Open', ...
    'High', ...
    'Low', ...
    'Close', ...
    'Volume', ...
    'Sentiment'};

for k = 1:numel(requiredColumns)

    if ~ismember( ...
            requiredColumns{k}, ...
            data.Properties.VariableNames)

        error('step3:missingColumn', ...
            'Required column "%s" is missing.', ...
            requiredColumns{k});

    end

end

%% -----------------------------------------------------------
% 6. DATE CONVERSION
% ------------------------------------------------------------

if ~isdatetime(data.Date)

    try

        data.Date = datetime(data.Date);

    catch ME

        error('step3:dateError', ...
            'Unable to convert Date to datetime.\n%s', ...
            ME.message);

    end

end

data = sortrows(data, 'Date');

%% -----------------------------------------------------------
% 7. CONVERT NUMERIC VARIABLES
% ------------------------------------------------------------

numericColumns = { ...
    'Open', ...
    'High', ...
    'Low', ...
    'Close', ...
    'Volume', ...
    'Sentiment'};

for k = 1:numel(numericColumns)

    colName = numericColumns{k};

    if ~isnumeric(data.(colName))

        data.(colName) = str2double( ...
            string(data.(colName)));

    end

end

%% -----------------------------------------------------------
% 8. BASIC DATA CLEANING
% ------------------------------------------------------------

validRows = ...
    ~isnat(data.Date) & ...
    isfinite(data.Open) & ...
    isfinite(data.High) & ...
    isfinite(data.Low) & ...
    isfinite(data.Close) & ...
    isfinite(data.Volume) & ...
    isfinite(data.Sentiment) & ...
    data.Close > 0;

removedRows = sum(~validRows);

if removedRows > 0

    fprintf('-> Removing %d invalid observations...\n', ...
        removedRows);

    data = data(validRows,:);

end

if height(data) < 100

    error('step3:insufficientData', ...
        'Insufficient observations for feature engineering.');

end

fprintf('   Valid observations: %d\n\n', ...
    height(data));

%% -----------------------------------------------------------
% 9. EXTRACT PRICE SERIES
% ------------------------------------------------------------

closePrice = data.Close;

highPrice = data.High;

lowPrice = data.Low;

volume = data.Volume;

sentiment = data.Sentiment;

n = height(data);

%% -----------------------------------------------------------
% 10. DAILY SIMPLE RETURN
% ------------------------------------------------------------

dailyReturn = nan(n,1);

dailyReturn(2:end) = ...
    closePrice(2:end) ./ ...
    closePrice(1:end-1) - 1;

%% -----------------------------------------------------------
% 11. LOG RETURN
% ------------------------------------------------------------

logReturn = nan(n,1);

logReturn(2:end) = ...
    log(closePrice(2:end) ./ ...
        closePrice(1:end-1));

%% -----------------------------------------------------------
% 12. LAGGED RETURNS
% ------------------------------------------------------------

returnLag1 = lagVector(dailyReturn, 1);

returnLag2 = lagVector(dailyReturn, 2);

returnLag3 = lagVector(dailyReturn, 3);

returnLag5 = lagVector(dailyReturn, 5);

returnLag7 = lagVector(dailyReturn, 7);

%% -----------------------------------------------------------
% 13. PRICE MOMENTUM
% ------------------------------------------------------------

momentum3 = nan(n,1);

momentum5 = nan(n,1);

momentum7 = nan(n,1);

momentum14 = nan(n,1);

momentum30 = nan(n,1);

momentum3(4:end) = ...
    closePrice(4:end) ./ ...
    closePrice(1:end-3) - 1;

momentum5(6:end) = ...
    closePrice(6:end) ./ ...
    closePrice(1:end-5) - 1;

momentum7(8:end) = ...
    closePrice(8:end) ./ ...
    closePrice(1:end-7) - 1;

momentum14(15:end) = ...
    closePrice(15:end) ./ ...
    closePrice(1:end-14) - 1;

momentum30(31:end) = ...
    closePrice(31:end) ./ ...
    closePrice(1:end-30) - 1;

%% -----------------------------------------------------------
% 14. SIMPLE MOVING AVERAGES
% ------------------------------------------------------------

sma5 = movmean( ...
    closePrice, ...
    [4 0], ...
    'omitnan');

sma10 = movmean( ...
    closePrice, ...
    [9 0], ...
    'omitnan');

sma20 = movmean( ...
    closePrice, ...
    [19 0], ...
    'omitnan');

sma50 = movmean( ...
    closePrice, ...
    [49 0], ...
    'omitnan');

%% -----------------------------------------------------------
% 15. EMA
% ------------------------------------------------------------

ema12 = calculateEMA( ...
    closePrice, ...
    12);

ema26 = calculateEMA( ...
    closePrice, ...
    26);

%% -----------------------------------------------------------
% 16. EMA DISTANCE FEATURES
% ------------------------------------------------------------

ema12Distance = ...
    closePrice ./ ema12 - 1;

ema26Distance = ...
    closePrice ./ ema26 - 1;

sma20Distance = ...
    closePrice ./ sma20 - 1;

sma50Distance = ...
    closePrice ./ sma50 - 1;

%% -----------------------------------------------------------
% 17. RSI
% ------------------------------------------------------------

rsi14 = calculateRSI( ...
    closePrice, ...
    14);

%% -----------------------------------------------------------
% 18. MACD
% ------------------------------------------------------------

macdLine = ema12 - ema26;

macdSignal = calculateEMA( ...
    macdLine, ...
    9);

macdHistogram = ...
    macdLine - macdSignal;

%% -----------------------------------------------------------
% 19. ROLLING VOLATILITY
% ------------------------------------------------------------

volatility5 = rollingStd( ...
    dailyReturn, ...
    5);

volatility10 = rollingStd( ...
    dailyReturn, ...
    10);

volatility20 = rollingStd( ...
    dailyReturn, ...
    20);

volatility30 = rollingStd( ...
    dailyReturn, ...
    30);

%% -----------------------------------------------------------
% 20. ATR
% ------------------------------------------------------------

previousClose = lagVector( ...
    closePrice, ...
    1);

trueRange = nan(n,1);

for i = 1:n

    if i == 1

        trueRange(i) = ...
            highPrice(i) - lowPrice(i);

    else

        trueRange(i) = max([ ...
            highPrice(i) - lowPrice(i), ...
            abs(highPrice(i) - previousClose(i)), ...
            abs(lowPrice(i) - previousClose(i))]);

    end

end

atr14 = movmean( ...
    trueRange, ...
    [13 0], ...
    'omitnan');

atr14Normalized = ...
    atr14 ./ closePrice;

%% -----------------------------------------------------------
% 21. HIGH-LOW RANGE
% ------------------------------------------------------------

dailyRange = ...
    (highPrice - lowPrice) ./ closePrice;

%% -----------------------------------------------------------
% 22. VOLUME FEATURES
% ------------------------------------------------------------

volumeLag1 = lagVector( ...
    volume, ...
    1);

volumeChange = nan(n,1);

validVolume = ...
    isfinite(volume) & ...
    isfinite(volumeLag1) & ...
    volumeLag1 > 0;

volumeChange(validVolume) = ...
    volume(validVolume) ./ ...
    volumeLag1(validVolume) - 1;

volumeSMA20 = movmean( ...
    volume, ...
    [19 0], ...
    'omitnan');

volumeRelative = ...
    volume ./ volumeSMA20;

%% -----------------------------------------------------------
% 23. SENTIMENT LAGS
% ------------------------------------------------------------

sentimentLag1 = lagVector( ...
    sentiment, ...
    1);

sentimentLag3 = lagVector( ...
    sentiment, ...
    3);

sentimentLag7 = lagVector( ...
    sentiment, ...
    7);

%% -----------------------------------------------------------
% 24. SENTIMENT MOVING AVERAGES
% ------------------------------------------------------------

sentimentMA3 = movmean( ...
    sentiment, ...
    [2 0], ...
    'omitnan');

sentimentMA7 = movmean( ...
    sentiment, ...
    [6 0], ...
    'omitnan');

sentimentMA14 = movmean( ...
    sentiment, ...
    [13 0], ...
    'omitnan');

sentimentChange = nan(n,1);

validSentimentChange = ...
    isfinite(sentiment) & ...
    isfinite(sentimentLag1);

sentimentChange(validSentimentChange) = ...
    sentiment(validSentimentChange) - ...
    sentimentLag1(validSentimentChange);

%% -----------------------------------------------------------
% 25. SENTIMENT REGIME
% ------------------------------------------------------------

if ismember( ...
        'SentimentRegime', ...
        data.Properties.VariableNames)

    sentimentRegime = data.SentimentRegime;

else

    % Reconstruct regime if necessary
    sentimentRegime = zeros(n,1);

    trainEnd = floor(0.70 * n);

    trainingSentiment = sentiment(1:trainEnd);

    lowerThreshold = ...
        prctile(trainingSentiment, 33.333333);

    upperThreshold = ...
        prctile(trainingSentiment, 66.666667);

    sentimentRegime( ...
        sentiment < lowerThreshold) = -1;

    sentimentRegime( ...
        sentiment > upperThreshold) = 1;

end

%% -----------------------------------------------------------
% 26. SENTIMENT × MOMENTUM INTERACTION
% ------------------------------------------------------------

sentimentReturnInteraction = ...
    sentimentLag1 .* returnLag1;

sentimentMomentumInteraction = ...
    sentimentLag1 .* momentum7;

%% -----------------------------------------------------------
% 27. ADD FEATURES TO TABLE
% ------------------------------------------------------------

data.Return = dailyReturn;

data.LogReturn = logReturn;

data.ReturnLag1 = returnLag1;

data.ReturnLag2 = returnLag2;

data.ReturnLag3 = returnLag3;

data.ReturnLag5 = returnLag5;

data.ReturnLag7 = returnLag7;

data.Momentum3 = momentum3;

data.Momentum5 = momentum5;

data.Momentum7 = momentum7;

data.Momentum14 = momentum14;

data.Momentum30 = momentum30;

data.SMA5 = sma5;

data.SMA10 = sma10;

data.SMA20 = sma20;

data.SMA50 = sma50;

data.EMA12 = ema12;

data.EMA26 = ema26;

data.EMA12Distance = ema12Distance;

data.EMA26Distance = ema26Distance;

data.SMA20Distance = sma20Distance;

data.SMA50Distance = sma50Distance;

data.RSI14 = rsi14;

data.MACD = macdLine;

data.MACDSignal = macdSignal;

data.MACDHistogram = macdHistogram;

data.Volatility5 = volatility5;

data.Volatility10 = volatility10;

data.Volatility20 = volatility20;

data.Volatility30 = volatility30;

data.ATR14 = atr14;

data.ATR14Normalized = atr14Normalized;

data.DailyRange = dailyRange;

data.VolumeChange = volumeChange;

data.VolumeSMA20 = volumeSMA20;

data.VolumeRelative = volumeRelative;

data.SentimentLag1 = sentimentLag1;

data.SentimentLag3 = sentimentLag3;

data.SentimentLag7 = sentimentLag7;

data.SentimentMA3 = sentimentMA3;

data.SentimentMA7 = sentimentMA7;

data.SentimentMA14 = sentimentMA14;

data.SentimentChange = sentimentChange;

data.SentimentRegime = sentimentRegime;

data.SentimentReturnInteraction = ...
    sentimentReturnInteraction;

data.SentimentMomentumInteraction = ...
    sentimentMomentumInteraction;

%% -----------------------------------------------------------
% 28. FUTURE RETURN TARGET
% ------------------------------------------------------------
%
% Target = next-day return.
%
% IMPORTANT:
% The target uses tomorrow's closing price and therefore must
% never be included as an input feature.

targetHorizon = 1;

futureReturn = nan(n,1);

if n > targetHorizon

    futureReturn(1:end-targetHorizon) = ...
        closePrice(1+targetHorizon:end) ./ ...
        closePrice(1:end-targetHorizon) - 1;

end

data.TargetReturn1D = futureReturn;

%% -----------------------------------------------------------
% 29. TARGET DIRECTION
% ------------------------------------------------------------

targetDirection = nan(n,1);

validTarget = isfinite(futureReturn);

targetDirection(validTarget) = ...
    double(futureReturn(validTarget) > 0);

data.TargetDirection = targetDirection;

%% -----------------------------------------------------------
% 30. REMOVE INF VALUES
% ------------------------------------------------------------

variableNames = data.Properties.VariableNames;

for k = 1:numel(variableNames)

    varName = variableNames{k};

    column = data.(varName);

    if isnumeric(column)

        bad = isinf(column);

        if any(bad)

            column(bad) = NaN;

            data.(varName) = column;

        end

    end

end

%% -----------------------------------------------------------
% 31. DISPLAY FEATURE INFORMATION
% ------------------------------------------------------------

fprintf('\n------------------------------------------------------------\n');
fprintf(' [FEATURE ENGINEERING SUMMARY]\n');
fprintf('------------------------------------------------------------\n');

fprintf(' Original observations : %d\n', n);

fprintf(' Total columns         : %d\n', width(data));

fprintf(' Engineered features   : %d\n', ...
    width(data) - 9);

fprintf(' Target horizon        : %d day\n', ...
    targetHorizon);

fprintf('\n');

fprintf(' Feature groups:\n');
fprintf('   Returns & lags\n');
fprintf('   Momentum\n');
fprintf('   Moving averages\n');
fprintf('   EMA\n');
fprintf('   RSI\n');
fprintf('   MACD\n');
fprintf('   Volatility\n');
fprintf('   ATR\n');
fprintf('   Volume\n');
fprintf('   Sentiment lags\n');
fprintf('   Sentiment moving averages\n');
fprintf('   Sentiment regimes\n');
fprintf('   Sentiment interactions\n');

fprintf('------------------------------------------------------------\n\n');

%% -----------------------------------------------------------
% 32. VALID TARGET COUNT
% ------------------------------------------------------------

fprintf('Valid target observations: %d / %d\n', ...
    sum(isfinite(data.TargetReturn1D)), ...
    height(data));

%% -----------------------------------------------------------
% 33. SAVE DATASET
% ------------------------------------------------------------

fprintf('\n-> Saving factor dataset...\n');
fprintf('   Output: %s\n', outputFile);

try

    writetable(data, outputFile);

catch ME

    error('step3:saveError', ...
        'Unable to save factor dataset.\n%s', ...
        ME.message);

end

%% -----------------------------------------------------------
% 34. CREATE FEATURE VISUALIZATION
% ------------------------------------------------------------

figureFile = fullfile( ...
    figureDir, ...
    'technical_and_sentiment_features.png');

fprintf('-> Generating feature visualization...\n');
fprintf('   Output: %s\n', figureFile);

try

    fig = figure( ...
        'Visible', 'off', ...
        'Position', [100 100 1200 700]);

    yyaxis left

    plot( ...
        data.Date, ...
        data.Close, ...
        'LineWidth', 1.2);

    ylabel('BTC Close Price');

    yyaxis right

    plot( ...
        data.Date, ...
        data.RSI14, ...
        'LineWidth', 1.0);

    ylabel('RSI(14)');

    xlabel('Date');

    title('BTC Price and RSI Feature');

    grid on;

    exportgraphics( ...
        fig, ...
        figureFile, ...
        'Resolution', 150);

    close(fig);

catch ME

    warning('step3:plotError', ...
        ['Feature plot could not be generated.\n' ...
         'Dataset was still saved.\n\n' ...
         'Original error: %s'], ...
        ME.message);

end

%% -----------------------------------------------------------
% 35. FINAL CHECK
% ------------------------------------------------------------

if ~isfile(outputFile)

    error('step3:outputMissing', ...
        'Factor dataset was not created.');

end

fprintf('\n============================================================\n');
fprintf(' [STEP 3 COMPLETE]\n');
fprintf('============================================================\n');

fprintf('Observations : %d\n', height(data));

fprintf('Columns      : %d\n', width(data));

fprintf('\nFactor dataset:\n%s\n', ...
    outputFile);

fprintf('\nFeature figure:\n%s\n', ...
    figureFile);

fprintf('\nProceeding to Step 4...\n\n');

end


%% ============================================================
function y = lagVector(x, lag)
%LAGVECTOR Shift a vector backward by lag observations.

x = x(:);

n = numel(x);

y = nan(n,1);

if lag < 0

    error('lagVector:invalidLag', ...
        'Lag must be non-negative.');

end

if lag == 0

    y = x;

    return;

end

if lag < n

    y(lag+1:end) = ...
        x(1:end-lag);

end

end


%% ============================================================
function ema = calculateEMA(x, period)
%CALCULATEEMA Calculate exponential moving average.

x = x(:);

n = numel(x);

ema = nan(n,1);

if period <= 0 || period ~= floor(period)

    error('calculateEMA:invalidPeriod', ...
        'EMA period must be a positive integer.');

end

if n == 0

    return;

end


alpha = 2 / (period + 1);

firstValid = find(isfinite(x), 1, 'first');

if isempty(firstValid)

    return;

end


ema(firstValid) = x(firstValid);


for i = firstValid+1:n

    if isfinite(x(i))

        if isfinite(ema(i-1))

            ema(i) = ...
                alpha * x(i) + ...
                (1-alpha) * ema(i-1);

        else

            ema(i) = x(i);

        end

    else

        ema(i) = ema(i-1);

    end

end

end


%% ============================================================
function rsi = calculateRSI(price, period)
%CALCULATERSI Calculate RSI using Wilder-style smoothing.

price = price(:);

n = numel(price);

rsi = nan(n,1);

if n <= period

    return;

end


delta = [NaN; diff(price)];

gain = max(delta, 0);

loss = max(-delta, 0);

gain(1) = NaN;

loss(1) = NaN;


% Initial Wilder averages
initialGain = mean( ...
    gain(2:period+1), ...
    'omitnan');

initialLoss = mean( ...
    loss(2:period+1), ...
    'omitnan');


avgGain = nan(n,1);

avgLoss = nan(n,1);

avgGain(period+1) = initialGain;

avgLoss(period+1) = initialLoss;


% Wilder smoothing
for i = period+2:n

    if isfinite(gain(i))

        avgGain(i) = ...
            ((period-1) * avgGain(i-1) + gain(i)) ...
            / period;

    end

    if isfinite(loss(i))

        avgLoss(i) = ...
            ((period-1) * avgLoss(i-1) + loss(i)) ...
            / period;

    end

end


% RSI
for i = period+1:n

    if ~isfinite(avgGain(i)) || ...
            ~isfinite(avgLoss(i))

        continue;

    end

    if avgLoss(i) == 0

        if avgGain(i) > 0
            rsi(i) = 100;
        else
            rsi(i) = 50;
        end

    else

        rs = avgGain(i) / avgLoss(i);

        rsi(i) = ...
            100 - 100 / (1 + rs);

    end

end

end


%% ============================================================
function s = rollingStd(x, window)
%ROLLINGSTD Rolling standard deviation using trailing window.

x = x(:);

n = numel(x);

s = nan(n,1);

for i = 1:n

    startIdx = max(1, i-window+1);

    values = x(startIdx:i);

    values = values(isfinite(values));

    if numel(values) >= 2

        s(i) = std(values, 0);

    end

end

end