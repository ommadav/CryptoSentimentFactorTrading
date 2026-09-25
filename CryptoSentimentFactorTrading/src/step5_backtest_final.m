%% ============================================================
% STEP 5 - FINAL CONSISTENT AR(5) BACKTEST
% Crypto Sentiment Factor Trading
%
% IMPORTANT:
% This file uses EXACTLY the same cleaning and chronological
% split methodology as the unified model comparison.
%
% Split:
%   Train      = 70%
%   Validation = 15%
%   Test       = remaining 15%
%
% AR(5) uses previous 5 target returns.
%
% Threshold is selected using VALIDATION only.
% Test data remains untouched until final evaluation.
%% ============================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       CONSISTENT FINAL AR(5) BACKTEST\n');
fprintf('============================================================\n\n');

%% ============================================================
% 1. PATHS
% ============================================================

srcFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(srcFolder);

dataFolder = fullfile(projectRoot,'data','processed');

resultsFolder = fullfile(projectRoot,'results');

tableFolder = fullfile(resultsFolder,'tables');
figureFolder = fullfile(resultsFolder,'figures');
modelFolder = fullfile(resultsFolder,'models');

if ~exist(tableFolder,'dir')
    mkdir(tableFolder);
end

if ~exist(figureFolder,'dir')
    mkdir(figureFolder);
end

if ~exist(modelFolder,'dir')
    mkdir(modelFolder);
end

dataFile = fullfile(dataFolder,'factor_dataset.csv');

%% ============================================================
% 2. LOAD DATA
% ============================================================

fprintf('Loading factor dataset...\n');

if ~isfile(dataFile)
    error('factor_dataset.csv not found:\n%s',dataFile);
end

T = readtable(dataFile);

fprintf('Original rows    : %d\n',height(T));
fprintf('Original columns : %d\n\n',width(T));

%% ============================================================
% 3. REQUIRED COLUMNS
% ============================================================

targetName = 'TargetReturn1D';
closeName  = 'Close';

if ~ismember(targetName,T.Properties.VariableNames)
    error('TargetReturn1D column not found.');
end

if ~ismember(closeName,T.Properties.VariableNames)
    error('Close column not found.');
end

%% ============================================================
% 4. FIND DATE COLUMN
% ============================================================

possibleDates = { ...
    'Date', ...
    'date', ...
    'OpenTime', ...
    'Open_time', ...
    'Open time'};

dateName = '';

for i = 1:numel(possibleDates)

    if ismember(possibleDates{i},T.Properties.VariableNames)
        dateName = possibleDates{i};
        break;
    end

end

if isempty(dateName)

    for i = 1:width(T)

        if isdatetime(T{:,i})
            dateName = T.Properties.VariableNames{i};
            break;
        end

    end

end

if isempty(dateName)
    error('Date column not found.');
end

%% ============================================================
% 5. CONVERT DATE
% ============================================================

dates = T.(dateName);

if ~isdatetime(dates)

    try
        dates = datetime(dates);

    catch

        try
            dates = datetime(dates,'InputFormat','yyyy-MM-dd');

        catch
            error('Unable to convert date column to datetime.');
        end

    end

end

%% ============================================================
% 6. TARGET AND CLOSE
% ============================================================

target = double(T.(targetName));
closePrice = double(T.(closeName));

%% ============================================================
% 7. SAME CLEANING AS UNIFIED MODEL COMPARISON
%
% First clean target/price/date.
% Then construct AR observations.
%
% The unified comparison removed invalid rows from the
% feature/target dataset before splitting.
%% ============================================================

validBase = ...
    isfinite(target) & ...
    isfinite(closePrice) & ...
    ~isnat(dates);

target = target(validBase);
closePrice = closePrice(validBase);
dates = dates(validBase);

fprintf('Target/price valid rows : %d\n',numel(target));

%% ============================================================
% 8. CREATE AR(5) OBSERVATIONS
% ============================================================

p = 5;

N = numel(target);

X = zeros(N-p,p);
Y = target(p+1:N);

D = dates(p+1:N);

P = closePrice(p+1:N);

for i = 1:(N-p)

    t = i+p;

    X(i,:) = target(t-1:-1:t-p)';

end

fprintf('AR(5) observations      : %d\n',size(X,1));

%% ============================================================
% 9. REMOVE INVALID AR FEATURES
% ============================================================

validAR = ...
    all(isfinite(X),2) & ...
    isfinite(Y) & ...
    isfinite(P);

X = X(validAR,:);
Y = Y(validAR);
D = D(validAR);
P = P(validAR);

fprintf('Final usable observations: %d\n\n',numel(Y));

%% ============================================================
% 10. CHRONOLOGICAL 70 / 15 / 15 SPLIT
% ============================================================

N = size(X,1);

nTrain = floor(0.70*N);
nVal   = floor(0.15*N);
nTest  = N-nTrain-nVal;

idxTrain = 1:nTrain;

idxVal = ...
    nTrain+1:nTrain+nVal;

idxTest = ...
    nTrain+nVal+1:N;

fprintf('------------------------------------------------------------\n');
fprintf('COMMON DATA SPLIT\n');
fprintf('------------------------------------------------------------\n');

fprintf('Train      : %d\n',numel(idxTrain));
fprintf('Validation : %d\n',numel(idxVal));
fprintf('Test       : %d\n\n',numel(idxTest));

fprintf('Train dates      : %s -> %s\n', ...
    datestr(D(idxTrain(1))), ...
    datestr(D(idxTrain(end))));

fprintf('Validation dates : %s -> %s\n', ...
    datestr(D(idxVal(1))), ...
    datestr(D(idxVal(end))));

fprintf('Test dates       : %s -> %s\n\n', ...
    datestr(D(idxTest(1))), ...
    datestr(D(idxTest(end))));

%% ============================================================
% 11. TRAINING DATA
% ============================================================

XTrain = X(idxTrain,:);
YTrain = Y(idxTrain);

%% ============================================================
% 12. STANDARDIZE TRAINING DATA
% ============================================================

muTrain = mean(XTrain,1);
sigmaTrain = std(XTrain,0,1);

sigmaTrain(sigmaTrain == 0) = 1;

XTrainZ = ...
    (XTrain-muTrain)./sigmaTrain;

%% ============================================================
% 13. RIDGE AR(5)
% ============================================================

lambda = 0.01;

XTrainZ = ...
    [ones(size(XTrainZ,1),1),XTrainZ];

I = eye(size(XTrainZ,2));

% Do not regularize intercept
I(1,1) = 0;

beta = ...
    (XTrainZ'*XTrainZ + lambda*I) ...
    \ ...
    (XTrainZ'*YTrain);

fprintf('AR(5) model trained.\n\n');

%% ============================================================
% 14. VALIDATION PREDICTIONS
% ============================================================

XVal = X(idxVal,:);
YVal = Y(idxVal);

XValZ = ...
    (XVal-muTrain)./sigmaTrain;

XValZ = ...
    [ones(size(XValZ,1),1),XValZ];

predVal = XValZ*beta;

%% ============================================================
% 15. VALIDATION METRICS
% ============================================================

valError = predVal-YVal;

valRMSE = sqrt(mean(valError.^2));

valMAE = mean(abs(valError));

SSres = sum((YVal-predVal).^2);
SStot = sum((YVal-mean(YVal)).^2);

if SStot > 0
    valR2 = 1-SSres/SStot;
else
    valR2 = NaN;
end

valDirection = sign(predVal)==sign(YVal);

valDirectionalAccuracy = mean(valDirection);

fprintf('------------------------------------------------------------\n');
fprintf('VALIDATION PERFORMANCE\n');
fprintf('------------------------------------------------------------\n');

fprintf('RMSE                 : %.6f\n',valRMSE);
fprintf('MAE                  : %.6f\n',valMAE);
fprintf('R2                   : %.6f\n',valR2);
fprintf('Directional Accuracy : %.2f%%\n\n', ...
    100*valDirectionalAccuracy);

%% ============================================================
% 16. THRESHOLD OPTIMIZATION
%
% IMPORTANT:
% Threshold is selected using validation only.
%% ============================================================

thresholdGrid = 0.001:0.001:0.020;

transactionCost = 0.001;

bestThreshold = thresholdGrid(1);
bestSharpe = -Inf;

for threshold = thresholdGrid

    signal = zeros(size(predVal));

    signal(predVal > threshold) = 1;
    signal(predVal < -threshold) = -1;

    grossReturn = signal.*YVal;

    turnover = zeros(size(signal));

    if ~isempty(signal)
        turnover(1) = abs(signal(1));
    end

    if numel(signal)>1
        turnover(2:end) = ...
            abs(diff(signal));
    end

    netReturn = ...
        grossReturn - ...
        transactionCost*turnover;

    if std(netReturn)>0

        sharpe = ...
            mean(netReturn)/std(netReturn)*sqrt(252);

    else

        sharpe = 0;

    end

    if isfinite(sharpe) && ...
            sharpe>bestSharpe

        bestSharpe = sharpe;
        bestThreshold = threshold;

    end

end

fprintf('------------------------------------------------------------\n');
fprintf('THRESHOLD OPTIMIZATION\n');
fprintf('------------------------------------------------------------\n');

fprintf('Best threshold : %.4f (%.2f%%)\n', ...
    bestThreshold, ...
    bestThreshold*100);

fprintf('Validation Sharpe : %.4f\n\n', ...
    bestSharpe);

%% ============================================================
% 17. REFIT USING TRAIN + VALIDATION
% ============================================================

fprintf('Refitting AR(5) using Train + Validation...\n');

idxDevelopment = ...
    [idxTrain,idxVal];

XDev = X(idxDevelopment,:);
YDev = Y(idxDevelopment);

muDev = mean(XDev,1);
sigmaDev = std(XDev,0,1);

sigmaDev(sigmaDev==0) = 1;

XDevZ = ...
    (XDev-muDev)./sigmaDev;

XDevZ = ...
    [ones(size(XDevZ,1),1),XDevZ];

I = eye(size(XDevZ,2));
I(1,1) = 0;

betaFinal = ...
    (XDevZ'*XDevZ + lambda*I) ...
    \ ...
    (XDevZ'*YDev);

%% ============================================================
% 18. TEST DATA
% ============================================================

XTest = X(idxTest,:);
YTest = Y(idxTest);

DTest = D(idxTest);
PTest = P(idxTest);

XTestZ = ...
    (XTest-muDev)./sigmaDev;

XTestZ = ...
    [ones(size(XTestZ,1),1),XTestZ];

predTest = XTestZ*betaFinal;

%% ============================================================
% 19. TEST PREDICTION METRICS
% ============================================================

testError = predTest-YTest;

testRMSE = sqrt(mean(testError.^2));

testMAE = mean(abs(testError));

SSres = sum((YTest-predTest).^2);
SStot = sum((YTest-mean(YTest)).^2);

if SStot>0
    testR2 = 1-SSres/SStot;
else
    testR2 = NaN;
end

testDirectionalAccuracy = ...
    mean(sign(predTest)==sign(YTest));

%% ============================================================
% 20. GENERATE TRADING SIGNALS
% ============================================================

signals = zeros(size(predTest));

signals(predTest>bestThreshold) = 1;

signals(predTest<-bestThreshold) = -1;

%% ============================================================
% 21. SIGNAL LABELS
% ============================================================

signalLabel = strings(size(signals));

signalLabel(signals==1) = "BUY";

signalLabel(signals==0) = "HOLD";

signalLabel(signals==-1) = "SELL";

%% ============================================================
% 22. BACKTEST
% ============================================================

initialCapital = 10000;

grossReturn = ...
    signals.*YTest;

turnover = zeros(size(signals));

if ~isempty(signals)
    turnover(1) = abs(signals(1));
end

if numel(signals)>1

    turnover(2:end) = ...
        abs(diff(signals));

end

transactionCosts = ...
    transactionCost*turnover;

netReturn = ...
    grossReturn-transactionCosts;

%% ============================================================
% 23. EQUITY CURVES
% ============================================================

strategyEquity = ...
    initialCapital*cumprod(1+netReturn);

buyHoldEquity = ...
    initialCapital*cumprod(1+YTest);

%% ============================================================
% 24. RETURNS
% ============================================================

strategyReturn = ...
    strategyEquity(end)/initialCapital-1;

buyHoldReturn = ...
    buyHoldEquity(end)/initialCapital-1;

%% ============================================================
% 25. SHARPE
% ============================================================

if std(netReturn)>0

    strategySharpe = ...
        mean(netReturn)/std(netReturn)*sqrt(252);

else

    strategySharpe = 0;

end

if std(YTest)>0

    buyHoldSharpe = ...
        mean(YTest)/std(YTest)*sqrt(252);

else

    buyHoldSharpe = 0;

end

%% ============================================================
% 26. MAX DRAWDOWN
% ============================================================

strategyPeak = cummax(strategyEquity);

strategyDrawdown = ...
    strategyEquity./strategyPeak-1;

strategyMaxDD = ...
    min(strategyDrawdown);

buyHoldPeak = cummax(buyHoldEquity);

buyHoldDrawdown = ...
    buyHoldEquity./buyHoldPeak-1;

buyHoldMaxDD = ...
    min(buyHoldDrawdown);

%% ============================================================
% 27. WIN RATE
% ============================================================

activeTrades = signals~=0;

if any(activeTrades)

    winRate = ...
        mean(netReturn(activeTrades)>0);

else

    winRate = NaN;

end

%% ============================================================
% 28. NUMBER OF TRADES
% ============================================================

numberOfTrades = ...
    sum(turnover>0);

%% ============================================================
% 29. PRINT FINAL RESULTS
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                  FINAL TEST RESULTS\n');
fprintf('============================================================\n');

fprintf('\nPREDICTION PERFORMANCE\n');
fprintf('------------------------------------------------------------\n');

fprintf('Test RMSE             : %.6f\n',testRMSE);

fprintf('Test MAE              : %.6f\n',testMAE);

fprintf('Test R2               : %.6f\n',testR2);

fprintf('Directional Accuracy  : %.2f%%\n', ...
    testDirectionalAccuracy*100);

fprintf('\nTRADING PARAMETERS\n');
fprintf('------------------------------------------------------------\n');

fprintf('Initial Capital       : $%.2f\n', ...
    initialCapital);

fprintf('Transaction Cost      : %.2f%%\n', ...
    transactionCost*100);

fprintf('Signal Threshold      : %.2f%%\n', ...
    bestThreshold*100);

fprintf('\nAR(5) STRATEGY\n');
fprintf('------------------------------------------------------------\n');

fprintf('Final Capital         : $%.2f\n', ...
    strategyEquity(end));

fprintf('Total Return          : %.2f%%\n', ...
    strategyReturn*100);

fprintf('Sharpe Ratio          : %.4f\n', ...
    strategySharpe);

fprintf('Maximum Drawdown      : %.2f%%\n', ...
    strategyMaxDD*100);

fprintf('Win Rate              : %.2f%%\n', ...
    winRate*100);

fprintf('Number of Trades      : %d\n', ...
    numberOfTrades);

fprintf('\nBUY & HOLD\n');
fprintf('------------------------------------------------------------\n');

fprintf('Final Capital         : $%.2f\n', ...
    buyHoldEquity(end));

fprintf('Total Return          : %.2f%%\n', ...
    buyHoldReturn*100);

fprintf('Sharpe Ratio          : %.4f\n', ...
    buyHoldSharpe);

fprintf('Maximum Drawdown      : %.2f%%\n', ...
    buyHoldMaxDD*100);

fprintf('\n============================================================\n');

%% ============================================================
% 30. SIGNAL DISTRIBUTION
% ============================================================

buyCount = sum(signals==1);
holdCount = sum(signals==0);
sellCount = sum(signals==-1);

fprintf('\nSIGNAL DISTRIBUTION\n');
fprintf('------------------------------------------------------------\n');

fprintf('BUY  : %d\n',buyCount);
fprintf('HOLD : %d\n',holdCount);
fprintf('SELL : %d\n',sellCount);

%% ============================================================
% 31. RESULT TABLE
% ============================================================

ResultTable = table( ...
    DTest, ...
    PTest, ...
    YTest, ...
    predTest, ...
    signalLabel, ...
    signals, ...
    grossReturn, ...
    transactionCosts, ...
    netReturn, ...
    strategyEquity, ...
    buyHoldEquity, ...
    'VariableNames',{ ...
    'Date', ...
    'Close', ...
    'ActualReturn', ...
    'PredictedReturn', ...
    'Signal', ...
    'Position', ...
    'GrossStrategyReturn', ...
    'TransactionCost', ...
    'NetStrategyReturn', ...
    'StrategyEquity', ...
    'BuyHoldEquity'});

%% ============================================================
% 32. SAVE RESULT TABLE
% ============================================================

outputCSV = ...
    fullfile(tableFolder,'final_consistent_ar5_backtest.csv');

writetable(ResultTable,outputCSV);

fprintf('\nBacktest table saved:\n%s\n',outputCSV);

%% ============================================================
% 33. SAVE MODEL
% ============================================================

modelInfo = struct();

modelInfo.modelType = 'AR(5) Ridge Regression';

modelInfo.lagOrder = p;

modelInfo.beta = betaFinal;

modelInfo.mu = muDev;

modelInfo.sigma = sigmaDev;

modelInfo.lambda = lambda;

modelInfo.threshold = bestThreshold;

modelInfo.transactionCost = transactionCost;

modelInfo.initialCapital = initialCapital;

modelInfo.trainObservations = numel(idxTrain);

modelInfo.validationObservations = numel(idxVal);

modelInfo.testObservations = numel(idxTest);

modelInfo.trainStart = D(idxTrain(1));

modelInfo.trainEnd = D(idxTrain(end));

modelInfo.validationStart = D(idxVal(1));

modelInfo.validationEnd = D(idxVal(end));

modelInfo.testStart = D(idxTest(1));

modelInfo.testEnd = D(idxTest(end));

modelInfo.validationRMSE = valRMSE;

modelInfo.validationMAE = valMAE;

modelInfo.validationR2 = valR2;

modelInfo.validationDirectionalAccuracy = ...
    valDirectionalAccuracy;

modelInfo.validationSharpe = ...
    bestSharpe;

modelInfo.testRMSE = testRMSE;

modelInfo.testMAE = testMAE;

modelInfo.testR2 = testR2;

modelInfo.testDirectionalAccuracy = ...
    testDirectionalAccuracy;

modelInfo.strategyReturn = ...
    strategyReturn;

modelInfo.strategySharpe = ...
    strategySharpe;

modelInfo.strategyMaxDrawdown = ...
    strategyMaxDD;

modelInfo.strategyWinRate = ...
    winRate;

modelInfo.numberOfTrades = ...
    numberOfTrades;

modelInfo.buyHoldReturn = ...
    buyHoldReturn;

modelInfo.buyHoldSharpe = ...
    buyHoldSharpe;

modelInfo.buyHoldMaxDrawdown = ...
    buyHoldMaxDD;

modelFile = ...
    fullfile(modelFolder,'final_consistent_ar5_model.mat');

save(modelFile,'modelInfo');

fprintf('Model saved:\n%s\n',modelFile);

%% ============================================================
% 34. FIGURE - EQUITY CURVE
% ============================================================

fig1 = figure( ...
    'Name','AR(5) vs Buy and Hold', ...
    'Color','w');

plot(DTest,strategyEquity,'LineWidth',2);

hold on;

plot(DTest,buyHoldEquity,'LineWidth',2);

grid on;

xlabel('Date');

ylabel('Portfolio Value ($)');

title('AR(5) Strategy vs Buy & Hold');

legend( ...
    {'AR(5) Strategy','Buy & Hold'}, ...
    'Location','best');

saveas(fig1, ...
    fullfile(figureFolder, ...
    'final_consistent_ar5_equity.png'));

%% ============================================================
% 35. FIGURE - PREDICTION
% ============================================================

fig2 = figure( ...
    'Name','AR(5) Prediction', ...
    'Color','w');

plot(DTest,YTest,'LineWidth',1.2);

hold on;

plot(DTest,predTest,'LineWidth',1.2);

yline(0,'--');

grid on;

xlabel('Date');

ylabel('Return');

title('AR(5) Actual vs Predicted Returns');

legend( ...
    {'Actual','Predicted','Zero'}, ...
    'Location','best');

saveas(fig2, ...
    fullfile(figureFolder, ...
    'final_consistent_ar5_prediction.png'));

%% ============================================================
% 36. FIGURE - DRAWDOWN
% ============================================================

fig3 = figure( ...
    'Name','AR(5) Drawdown', ...
    'Color','w');

plot(DTest,strategyDrawdown*100,'LineWidth',1.5);

hold on;

plot(DTest,buyHoldDrawdown*100,'LineWidth',1.5);

grid on;

xlabel('Date');

ylabel('Drawdown (%)');

title('AR(5) Strategy vs Buy & Hold Drawdown');

legend( ...
    {'AR(5)','Buy & Hold'}, ...
    'Location','best');

saveas(fig3, ...
    fullfile(figureFolder, ...
    'final_consistent_ar5_drawdown.png'));

%% ============================================================
% 37. FIGURE - SIGNALS
% ============================================================

fig4 = figure( ...
    'Name','AR(5) Trading Signals', ...
    'Color','w');

plot(DTest,PTest,'LineWidth',1.5);

hold on;

buyIdx = signals==1;
sellIdx = signals==-1;

if any(buyIdx)

    scatter( ...
        DTest(buyIdx), ...
        PTest(buyIdx), ...
        50,'^','filled');

end

if any(sellIdx)

    scatter( ...
        DTest(sellIdx), ...
        PTest(sellIdx), ...
        50,'v','filled');

end

grid on;

xlabel('Date');

ylabel('BTC Close Price ($)');

title('AR(5) BUY / SELL Signals');

legend( ...
    {'BTC Price','BUY','SELL'}, ...
    'Location','best');

saveas(fig4, ...
    fullfile(figureFolder, ...
    'final_consistent_ar5_signals.png'));

%% ============================================================
% 38. FINAL SUMMARY TABLE
% ============================================================

SummaryTable = table( ...
    ["AR(5)";"Buy & Hold"], ...
    [strategyReturn;buyHoldReturn]*100, ...
    [strategySharpe;buyHoldSharpe], ...
    [strategyMaxDD;buyHoldMaxDD]*100, ...
    [strategyEquity(end);buyHoldEquity(end)], ...
    'VariableNames',{ ...
    'Strategy', ...
    'ReturnPercent', ...
    'Sharpe', ...
    'MaxDrawdownPercent', ...
    'FinalCapital'});

summaryCSV = ...
    fullfile(tableFolder, ...
    'final_consistent_backtest_summary.csv');

writetable(SummaryTable,summaryCSV);

fprintf('\nSummary saved:\n%s\n',summaryCSV);

%% ============================================================
% 39. COMPLETION
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('        CONSISTENT AR(5) BACKTEST COMPLETED\n');
fprintf('============================================================\n');

fprintf('\nFiles generated:\n');

fprintf('1. final_consistent_ar5_backtest.csv\n');

fprintf('2. final_consistent_backtest_summary.csv\n');

fprintf('3. final_consistent_ar5_model.mat\n');

fprintf('4. final_consistent_ar5_equity.png\n');

fprintf('5. final_consistent_ar5_prediction.png\n');

fprintf('6. final_consistent_ar5_drawdown.png\n');

fprintf('7. final_consistent_ar5_signals.png\n');

fprintf('\n============================================================\n');