function b = backtestUtils()
%BACKTESTUTILS Quantitative algorithmic trading strategy backtesting engine.
%Computes equity curves, transaction costs, drawdown, risk metrics (Sharpe, Sortino,
%Calmar, VaR, CVaR), trade statistics and signal generation.
%
%   B = BACKTESTUTILS() returns a struct of function handles.

b.generateSignals       = @generateSignals;
b.optimizeThreshold     = @optimizeThreshold;
b.runBacktest           = @runBacktest;
b.computeRiskMetrics    = @computeRiskMetrics;
b.compareStrategies     = @compareStrategies;

end

%% ------------------------------------------------------------
function signals = generateSignals(predictedReturns, buyThreshold, sellThreshold)
%GENERATESIGNALS Converts predicted returns to categorical trading signals.
%   BUY  when predictedReturn > buyThreshold
%   SELL when predictedReturn < -sellThreshold
%   HOLD otherwise

if nargin < 2 || isempty(buyThreshold), buyThreshold = 0.002; end
if nargin < 3 || isempty(sellThreshold), sellThreshold = 0.002; end

signals = repmat("HOLD", size(predictedReturns));
signals(predictedReturns > buyThreshold) = "BUY";
signals(predictedReturns < -sellThreshold) = "SELL";

end

%% ------------------------------------------------------------
function [bestBuyThresh, bestSellThresh, bestSharpe] = optimizeThreshold(valPreds, valPrices, feeRate)
%OPTIMIZETHRESHOLD Performs grid search on validation split to find optimal threshold.
if nargin < 3, feeRate = 0.001; end

candidateThresh = linspace(0.000, 0.015, 16);
bestSharpe = -Inf;
bestBuyThresh = 0.002;
bestSellThresh = 0.002;

for i = 1:numel(candidateThresh)
    th = candidateThresh(i);
    sigs = generateSignals(valPreds, th, th);
    bt = runBacktest(valPrices, sigs, 100, feeRate, 'LongShort');
    if bt.sharpeRatio > bestSharpe
        bestSharpe = bt.sharpeRatio;
        bestBuyThresh = th;
        bestSellThresh = th;
    end
end
end

%% ------------------------------------------------------------
function result = runBacktest(prices, signals, initialCapital, feeRate, strategyMode)
%RUNBACKTEST Simulates portfolio execution over time with friction / transaction costs.
%
%   strategyMode: 'LongShort' (BUY=+1, SELL=-1), 'LongOnly' (BUY=+1, SELL=0),
%                 or 'SentimentFilter'

if nargin < 3 || isempty(initialCapital), initialCapital = 10000; end
if nargin < 4 || isempty(feeRate), feeRate = 0.001; end % 10 bps default
if nargin < 5 || isempty(strategyMode), strategyMode = 'LongShort'; end

nDays = length(prices);
capital = initialCapital;
equity = zeros(nDays, 1);
equity(1) = initialCapital;

position = 0; % -1, 0, +1
positions = zeros(nDays, 1);
tradeCount = 0;
dailyReturn = zeros(nDays - 1, 1);
winningDays = 0;
losingDays = 0;
grossProfits = 0;
grossLosses = 0;

for t = 1:(nDays - 1)
    sig = signals(t);
    
    % Determine target position
    if strcmp(strategyMode, 'LongOnly')
        if strcmp(sig, 'BUY')
            newPos = 1;
        else
            newPos = 0;
        end
    else % LongShort
        if strcmp(sig, 'BUY')
            newPos = 1;
        elseif strcmp(sig, 'SELL')
            newPos = -1;
        else
            newPos = 0;
        end
    end
    
    % Calculate transaction fee on position change
    posChange = abs(newPos - position);
    fee = posChange * feeRate * capital;
    capital = capital - fee;
    
    if posChange > 0
        tradeCount = tradeCount + 1;
    end
    position = newPos;
    positions(t) = position;
    
    % Asset price return from day t to t+1
    marketReturn = (prices(t+1) - prices(t)) / prices(t);
    strategyRet = position * marketReturn;
    dailyReturn(t) = strategyRet;
    
    capital = capital * (1 + strategyRet);
    equity(t+1) = capital;
    
    if strategyRet > 0
        winningDays = winningDays + 1;
        grossProfits = grossProfits + (capital * strategyRet);
    elseif strategyRet < 0
        losingDays = losingDays + 1;
        grossLosses = grossLosses + abs(capital * strategyRet);
    end
end
positions(end) = position;

% Performance Metrics
totalReturnPct = 100 * (capital / initialCapital - 1);
buyHoldReturnPct = 100 * (prices(end) / prices(1) - 1);
annualizedReturnPct = 100 * ((capital / initialCapital) ^ (365 / max(nDays, 1)) - 1);

% Volatility & Sharpe
stdRet = std(dailyReturn);
if stdRet > 1e-6
    sharpeRatio = (mean(dailyReturn) / stdRet) * sqrt(365);
else
    sharpeRatio = 0;
end

% Sortino Ratio (Downside deviation only)
downsideRet = dailyReturn(dailyReturn < 0);
if ~isempty(downsideRet) && std(downsideRet) > 1e-6
    sortinoRatio = (mean(dailyReturn) / std(downsideRet)) * sqrt(365);
else
    sortinoRatio = sharpeRatio;
end

% Drawdown
runningMax = cummax(equity);
drawdown = (equity - runningMax) ./ max(runningMax, eps);
maxDrawdownPct = 100 * min(drawdown);

% Calmar Ratio
if abs(maxDrawdownPct) > 1e-4
    calmarRatio = annualizedReturnPct / abs(maxDrawdownPct);
else
    calmarRatio = 0;
end

% Win Rate & Profit Factor
totTradeDays = winningDays + losingDays;
if totTradeDays > 0
    winRatePct = 100 * (winningDays / totTradeDays);
else
    winRatePct = 0;
end

if grossLosses > 0
    profitFactor = grossProfits / grossLosses;
else
    profitFactor = 1.0;
end

% Value at Risk (VaR 95%) & CVaR (Expected Shortfall)
sortedRets = sort(dailyReturn);
var95Idx = max(1, floor(0.05 * length(sortedRets)));
var95Pct = 100 * abs(sortedRets(var95Idx));
cvar95Pct = 100 * abs(mean(sortedRets(1:var95Idx)));

result = struct();
result.equity              = equity;
result.dailyReturn         = dailyReturn;
result.drawdown            = drawdown;
result.positions           = positions;
result.finalCapital        = capital;
result.initialCapital      = initialCapital;
result.totalReturnPct      = totalReturnPct;
result.buyHoldReturnPct    = buyHoldReturnPct;
result.annualizedReturnPct = annualizedReturnPct;
result.sharpeRatio         = sharpeRatio;
result.sortinoRatio        = sortinoRatio;
result.maxDrawdownPct      = maxDrawdownPct;
result.calmarRatio         = calmarRatio;
result.winRatePct          = winRatePct;
result.profitFactor        = profitFactor;
result.tradeCount          = tradeCount;
result.var95Pct            = var95Pct;
result.cvar95Pct           = cvar95Pct;
result.strategyMode        = strategyMode;

end

%% ------------------------------------------------------------
function risk = computeRiskMetrics(dailyReturns)
%COMPUTERISKMETRICS Computes comprehensive standalone risk metrics.

stdDaily = std(dailyReturns);
risk = struct();
risk.volatilityAnnualized = stdDaily * sqrt(365) * 100;
risk.skewness = skewness(dailyReturns);
risk.kurtosis = kurtosis(dailyReturns);

sortedR = sort(dailyReturns);
n = length(sortedR);
risk.var95 = abs(sortedR(max(1, floor(0.05 * n)))) * 100;
risk.var99 = abs(sortedR(max(1, floor(0.01 * n)))) * 100;
risk.cvar95 = abs(mean(sortedR(1:max(1, floor(0.05 * n))))) * 100;

end

%% ------------------------------------------------------------
function compTable = compareStrategies(resultsMap)
%COMPARESTRATEGIES Formats a comparison table across multiple strategies.

names = fieldnames(resultsMap);
n = numel(names);

StrategyName   = cell(n, 1);
TotalReturn    = zeros(n, 1);
AnnualReturn   = zeros(n, 1);
Sharpe         = zeros(n, 1);
Sortino        = zeros(n, 1);
MaxDrawdown    = zeros(n, 1);
WinRate        = zeros(n, 1);
Trades         = zeros(n, 1);

for i = 1:n
    s = resultsMap.(names{i});
    StrategyName{i} = names{i};
    TotalReturn(i)  = s.totalReturnPct;
    AnnualReturn(i) = s.annualizedReturnPct;
    Sharpe(i)       = s.sharpeRatio;
    Sortino(i)      = s.sortinoRatio;
    MaxDrawdown(i)  = s.maxDrawdownPct;
    WinRate(i)      = s.winRatePct;
    Trades(i)       = s.tradeCount;
end

compTable = table(StrategyName, TotalReturn, AnnualReturn, Sharpe, Sortino, MaxDrawdown, WinRate, Trades);
end
