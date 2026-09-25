function f = featureUtils()
%FEATUREUTILS Comprehensive technical and sentiment feature engineering.
%
%   F = FEATUREUTILS() returns a struct of function handles.

f.engineerAllFeatures  = @engineerAllFeatures;
f.createReturnTarget   = @createReturnTarget;
f.normalizeFeatures    = @normalizeFeatures;
f.applyNormalization   = @applyNormalization;
f.computeWilderSmooth  = @computeWilderSmooth;
f.computeEMA           = @computeEMA;
f.computeRSI           = @computeRSI;
f.computeMACD          = @computeMACD;
f.computeBollingerBands= @computeBollingerBands;

end

%% ------------------------------------------------------------
function out = computeWilderSmooth(x, period)
%COMPUTEWILDERSMOOTH Wilder's smoothing for RSI.
n = length(x);
out = NaN(n, 1);
if n < period
    return;
end
out(period) = mean(x(1:period), 'omitnan');
for t = period+1:n
    if isnan(x(t))
        out(t) = out(t-1);
    else
        out(t) = (out(t-1) * (period - 1) + x(t)) / period;
    end
end
end

%% ------------------------------------------------------------
function out = computeEMA(x, period)
%COMPUTEEMA Exponential moving average.
n = length(x);
out = NaN(n, 1);
if n < period
    return;
end
alpha = 2 / (period + 1);
out(period) = mean(x(1:period), 'omitnan');
for t = period+1:n
    if isnan(x(t))
        out(t) = out(t-1);
    else
        out(t) = alpha * x(t) + (1 - alpha) * out(t-1);
    end
end
end

%% ------------------------------------------------------------
function rsi = computeRSI(price, period)
%COMPUTERSI Relative Strength Index (0 to 100).
if nargin < 2, period = 14; end
delta = [NaN; diff(price)];
gain = max(delta, 0);
loss = max(-delta, 0);
avgGain = computeWilderSmooth(gain, period);
avgLoss = computeWilderSmooth(loss, period);
rs = avgGain ./ max(avgLoss, eps);
rsi = 100 - (100 ./ (1 + rs));
end

%% ------------------------------------------------------------
function [macdLine, signalLine, hist] = computeMACD(price, fastPeriod, slowPeriod, signalPeriod)
%COMPUTEMACD Moving Average Convergence Divergence.
if nargin < 2, fastPeriod = 12; end
if nargin < 3, slowPeriod = 26; end
if nargin < 4, signalPeriod = 9; end

fastEMA = computeEMA(price, fastPeriod);
slowEMA = computeEMA(price, slowPeriod);
macdLine = fastEMA - slowEMA;
signalLine = computeEMA(macdLine, signalPeriod);
hist = macdLine - signalLine;
end

%% ------------------------------------------------------------
function [upper, middle, lower, bandwidth] = computeBollingerBands(price, period, numStd)
%COMPUTEBOLLINGERBANDS Bollinger Bands and normalized Bandwidth.
if nargin < 2, period = 20; end
if nargin < 3, numStd = 2; end

middle = movmean(price, [period-1 0], 'omitnan');
sigma  = movstd(price, [period-1 0], 'omitnan');
upper  = middle + numStd * sigma;
lower  = middle - numStd * sigma;
bandwidth = (upper - lower) ./ max(middle, eps);
end

%% ------------------------------------------------------------
function [X, factorNames, factorCategories] = engineerAllFeatures(mergedTable)
%ENGINEERALLFEATURES Generates a rich, multi-domain factor matrix including:
%   - Price / Momentum Factors (6)
%   - Volatility & Risk Factors (3)
%   - Technical Oscillators (4)
%   - Volume Factors (3)
%   - Sentiment & Interaction Factors (6)
% Total: 22 engineered features.

price     = double(mergedTable.Close);
volume    = double(mergedTable.Volume);
sentiment = double(mergedTable.Sentiment);
N         = length(price);

% 1. Price / Momentum
ret1D  = [NaN; diff(price) ./ price(1:end-1)];
ret3D  = [NaN(3,1); (price(4:end) - price(1:end-3)) ./ price(1:end-3)];
ret7D  = [NaN(7,1); (price(8:end) - price(1:end-7)) ./ price(1:end-7)];
ret14D = [NaN(14,1); (price(15:end) - price(1:end-14)) ./ price(1:end-14)];

sma14 = movmean(price, [13 0], 'omitnan');
sma50 = movmean(price, [49 0], 'omitnan');
priceVsSMA14 = (price ./ max(sma14, eps)) - 1;
priceVsSMA50 = (price ./ max(sma50, eps)) - 1;

% 2. Volatility
vol7  = movstd(ret1D, [6 0], 'omitnan');
vol14 = movstd(ret1D, [13 0], 'omitnan');
vol30 = movstd(ret1D, [29 0], 'omitnan');

% 3. Technical
rsi14 = computeRSI(price, 14);
[macdLine, signalLine, ~] = computeMACD(price, 12, 26, 9);
[~, ~, ~, bbWidth] = computeBollingerBands(price, 20, 2);

% 4. Volume
volChange = [NaN; diff(volume) ./ max(volume(1:end-1), eps)];
volMA7    = movmean(volume, [6 0], 'omitnan');
volMA30   = movmean(volume, [29 0], 'omitnan');
relVol7   = volume ./ max(volMA7, eps);
volRatio  = volMA7 ./ max(volMA30, eps);

% 5. Sentiment & Interactions
sentChange = [NaN; diff(sentiment)];
sentMA3    = movmean(sentiment, [2 0], 'omitnan');
sentMA7    = movmean(sentiment, [6 0], 'omitnan');
sentVol7   = movstd(sentiment, [6 0], 'omitnan');
sentRetInteract = sentiment .* ret1D;

% Assemble Feature Matrix
X = [ ...
    ret1D, ret3D, ret7D, ret14D, priceVsSMA14, priceVsSMA50, ...
    vol7, vol14, vol30, ...
    rsi14, macdLine, signalLine, bbWidth, ...
    volChange, relVol7, volRatio, ...
    sentiment, sentChange, sentMA3, sentMA7, sentVol7, sentRetInteract ...
];

factorNames = { ...
    'Return1D', 'Return3D', 'Return7D', 'Return14D', 'PriceVsSMA14', 'PriceVsSMA50', ...
    'Volatility7', 'Volatility14', 'Volatility30', ...
    'RSI14', 'MACD', 'MACDSignal', 'BBWidth', ...
    'VolumeChange', 'RelativeVolume7', 'VolumeRatio7_30', ...
    'Sentiment', 'SentimentChange', 'SentimentMA3', 'SentimentMA7', 'SentimentVolatility7', 'SentimentReturnInteraction' ...
}';

factorCategories = { ...
    'Momentum', 'Momentum', 'Momentum', 'Momentum', 'Momentum', 'Momentum', ...
    'Volatility', 'Volatility', 'Volatility', ...
    'Technical', 'Technical', 'Technical', 'Technical', ...
    'Volume', 'Volume', 'Volume', ...
    'Sentiment', 'Sentiment', 'Sentiment', 'Sentiment', 'Sentiment', 'Sentiment' ...
}';

assert(size(X, 2) == numel(factorNames), 'Feature count mismatch.');
assert(size(X, 1) == N, 'Row count mismatch.');

end

%% ------------------------------------------------------------
function [y, validMask] = createReturnTarget(price, horizon)
%CREATERETURNTARGET Computes forward returns for target prediction.
if nargin < 2, horizon = 1; end
N = length(price);
y = NaN(N, 1);
y(1:N-horizon) = (price(1+horizon:N) - price(1:N-horizon)) ./ price(1:N-horizon);
validMask = ~isnan(y);
end

%% ------------------------------------------------------------
function [mu, sigma] = normalizeFeatures(XTrain)
%NORMALIZEFEATURES Computes mean and standard deviation from training set.
mu = mean(XTrain, 1, 'omitnan');
sigma = std(XTrain, 0, 1, 'omitnan');
sigma(isnan(sigma) | sigma < 1e-12) = 1;
end

%% ------------------------------------------------------------
function Xnorm = applyNormalization(X, mu, sigma)
%APPLYNORMALIZATION Applies training z-score normalization to any split.
Xnorm = (X - mu) ./ sigma;
end
