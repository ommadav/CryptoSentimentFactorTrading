function step5_final_master_backtest()
%STEP5_FINAL_MASTER_BACKTEST Rigorous AR(5) research backtest.
%
% Improvements:
%   1. No artificial initialization of AR(5) lags.
%   2. Chronological data handling.
%   3. Target = next-day return.
%   4. Transaction costs based on position turnover.
%   5. Threshold selected using validation data only.
%   6. Final model refit on train + validation.
%   7. Completely untouched test set.
%
% Model:
%   Ridge-regularized AR(5):
%
%   r(t) = b0 + b1*r(t-1) + ... + b5*r(t-5) + error
%
% Trading:
%   prediction >= +threshold -> BUY
%   prediction <= -threshold -> SELL
%   otherwise                -> HOLD
%
% Academic historical backtest only.
% It does not model all real-world effects such as spread,
% slippage, funding, latency, and market impact.

clc;
close all;

fprintf('\n============================================================\n');
fprintf(' [STEP 5] RIGOROUS FINAL MASTER AR(5) BACKTEST\n');
fprintf('============================================================\n\n');

%% ========================================================================
% 1. PATHS
% =========================================================================

thisFile = mfilename('fullpath');

if isempty(thisFile)
    srcDir = pwd;
else
    srcDir = fileparts(thisFile);
end

projectDir = fileparts(srcDir);

dataFile = fullfile( ...
    projectDir, ...
    'data', ...
    'processed', ...
    'merged_dataset.csv');

if ~isfile(dataFile)

    error('step5:missingData', ...
        ['Missing merged_dataset.csv. ' ...
         'Run step1_data_loading_and_eda.m first.']);

end

resultsDir = fullfile(projectDir,'results');
tablesDir  = fullfile(resultsDir,'tables');
modelsDir  = fullfile(resultsDir,'models');
figDir     = fullfile(resultsDir,'figures');

if ~exist(tablesDir,'dir')
    mkdir(tablesDir);
end

if ~exist(modelsDir,'dir')
    mkdir(modelsDir);
end

if ~exist(figDir,'dir')
    mkdir(figDir);
end

%% ========================================================================
% 2. CONFIGURATION
% =========================================================================

initialCapital = 10000;

% 0.10% transaction cost
feeRate = 0.0010;

lagOrder = 5;

% Predict one day ahead
predictionHorizon = 1;

% Chronological split
trainRatio = 0.70;
valRatio   = 0.15;
testRatio  = 0.15;

if abs(trainRatio + valRatio + testRatio - 1) > 1e-12

    error('step5:badSplit', ...
        'Train/validation/test ratios must sum to 1.');

end

% Threshold candidates.
%
% These thresholds are evaluated ONLY on validation data.
%
% 0.20% -> 2.00%
thresholdGrid = (0.002:0.001:0.020)';

fprintf('Initial capital      : $%.2f\n',initialCapital);
fprintf('Transaction cost     : %.2f%%\n',feeRate*100);
fprintf('AR order             : %d\n',lagOrder);
fprintf('Prediction horizon   : %d day\n',predictionHorizon);
fprintf('Split                : %.0f%% / %.0f%% / %.0f%%\n\n', ...
    trainRatio*100, ...
    valRatio*100, ...
    testRatio*100);

%% ========================================================================
% 3. LOAD DATA
% =========================================================================

T = readtable( ...
    dataFile, ...
    'VariableNamingRule','preserve');

dateCol = findVariable(T, ...
    {'Date','date','Open time','Open_time','time','timestamp'});

closeCol = findVariable(T, ...
    {'Close','close'});

sentCol = findVariable(T, ...
    {'Sentiment','sentiment'});

if isempty(dateCol)

    error('step5:missingDate', ...
        'Could not identify the Date column.');

end

if isempty(closeCol)

    error('step5:missingClose', ...
        'Could not identify the Close column.');

end

dates = parseDateColumn(T{:,dateCol});

prices = toNumeric(T{:,closeCol});

if ~isempty(sentCol)

    sentiment = toNumeric(T{:,sentCol});

else

    sentiment = NaN(height(T),1);

end

%% ========================================================================
% 4. BASIC CLEANING
% =========================================================================

valid = ...
    ~isnat(dates) & ...
    isfinite(prices) & ...
    prices > 0;

dates = dates(valid);

prices = prices(valid);

sentiment = sentiment(valid);

% Sort chronologically
[dates,sortIdx] = sort(dates);

prices = prices(sortIdx);

sentiment = sentiment(sortIdx);

% Remove duplicate dates
[~,keepIdx] = unique(dates,'last');

keepIdx = sort(keepIdx);

dates = dates(keepIdx);

prices = prices(keepIdx);

sentiment = sentiment(keepIdx);

Nraw = numel(prices);

if Nraw < 100

    error('step5:tooLittleData', ...
        'Too few valid observations after cleaning.');

end

fprintf('Raw valid observations : %d\n',Nraw);

fprintf('Date range             : %s -> %s\n\n', ...
    datestr(dates(1),'yyyy-mm-dd'), ...
    datestr(dates(end),'yyyy-mm-dd'));

%% ========================================================================
% 5. CREATE NEXT-DAY TARGET
% =========================================================================
%
% target(t) =
%
%       Close(t+1)
%       ----------- - 1
%        Close(t)
%
% Therefore:
%
% Information available at day t
%              |
%              v
%        AR(5) prediction
%              |
%              v
% Actual return from t -> t+1
%
% This avoids look-ahead.

target = NaN(Nraw,1);

target(1:Nraw-predictionHorizon) = ...
    ( ...
    prices(1+predictionHorizon:Nraw) ...
    - prices(1:Nraw-predictionHorizon) ...
    ) ...
    ./ prices(1:Nraw-predictionHorizon);

%% ========================================================================
% 6. CREATE TRUE AR(5) LAGS
% =========================================================================
%
% We do NOT fill missing lag values artificially.
%
% For every row:
%
% X(:,1) = return(t-1)
% X(:,2) = return(t-2)
% X(:,3) = return(t-3)
% X(:,4) = return(t-4)
% X(:,5) = return(t-5)

X = NaN(Nraw,lagOrder);

for lag = 1:lagOrder

    X(:,lag) = ...
        [ ...
        NaN(lag,1); ...
        target(1:Nraw-lag) ...
        ];

end

% Only retain observations where:
%
%   target exists
%   all five AR lags exist

validModel = ...
    isfinite(target) & ...
    all(isfinite(X),2);

X = X(validModel,:);

y = target(validModel);

modelDates = dates(validModel);

modelPrices = prices(validModel);

modelSentiment = sentiment(validModel);

N = numel(y);

if N < 100

    error('step5:tooLittleModelData', ...
        'Too few observations after creating AR(%d) lags.',lagOrder);

end

fprintf('Valid AR(%d) observations : %d\n',lagOrder,N);

fprintf('Warm-up/terminal rows removed : %d\n\n', ...
    Nraw-N);

%% ========================================================================
% 7. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
% =========================================================================

nTrain = floor(trainRatio*N);

nVal = floor(valRatio*N);

nTest = N - nTrain - nVal;

if min([nTrain,nVal,nTest]) < 20

    error('step5:badSplit', ...
        'One of the splits is too small.');

end

trainIdx = 1:nTrain;

valIdx = ...
    (nTrain+1):(nTrain+nVal);

testIdx = ...
    (nTrain+nVal+1):N;

fprintf('------------------------------------------------------------\n');
fprintf(' CHRONOLOGICAL DATA SPLIT\n');
fprintf('------------------------------------------------------------\n');

fprintf('Train      : %4d | %s -> %s\n', ...
    numel(trainIdx), ...
    datestr(modelDates(trainIdx(1)),'yyyy-mm-dd'), ...
    datestr(modelDates(trainIdx(end)),'yyyy-mm-dd'));

fprintf('Validation : %4d | %s -> %s\n', ...
    numel(valIdx), ...
    datestr(modelDates(valIdx(1)),'yyyy-mm-dd'), ...
    datestr(modelDates(valIdx(end)),'yyyy-mm-dd'));

fprintf('Test       : %4d | %s -> %s\n', ...
    numel(testIdx), ...
    datestr(modelDates(testIdx(1)),'yyyy-mm-dd'), ...
    datestr(modelDates(testIdx(end)),'yyyy-mm-dd'));

fprintf('------------------------------------------------------------\n\n');

%% ========================================================================
% 8. TRAIN DEVELOPMENT AR(5)
% =========================================================================

% Standardize using TRAINING data only.

muX = mean(X(trainIdx,:),1,'omitnan');

sigmaX = std(X(trainIdx,:),0,1,'omitnan');

sigmaX( ...
    ~isfinite(sigmaX) | sigmaX < 1e-12 ...
    ) = 1;

XtrainN = ...
    (X(trainIdx,:) - muX) ./ sigmaX;

XvalN = ...
    (X(valIdx,:) - muX) ./ sigmaX;

XtestN = ...
    (X(testIdx,:) - muX) ./ sigmaX;

yTrain = y(trainIdx);

yVal = y(valIdx);

yTest = y(testIdx);

% Ridge regularization
ridgeLambda = 1e-4;

modelDev = fitrlinear( ...
    XtrainN, ...
    yTrain, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda',ridgeLambda, ...
    'Solver','lbfgs');

%% ========================================================================
% 9. VALIDATION PREDICTIONS
% =========================================================================

predVal = predict(modelDev,XvalN);

%% ========================================================================
% 10. VALIDATION METRICS
% =========================================================================

valRMSE = rmseMetric( ...
    yVal, ...
    predVal);

valMAE = ...
    mean(abs(yVal-predVal),'omitnan');

valR2 = ...
    r2Metric(yVal,predVal);

valDir = ...
    directionalAccuracy(yVal,predVal);

fprintf('DEVELOPMENT MODEL - VALIDATION\n');

fprintf('RMSE                 : %.6f\n',valRMSE);

fprintf('MAE                  : %.6f\n',valMAE);

fprintf('R2                   : %.6f\n',valR2);

fprintf('Directional accuracy : %.2f%%\n\n', ...
    valDir*100);

%% ========================================================================
% 11. VALIDATION-ONLY THRESHOLD OPTIMIZATION
% =========================================================================
%
% IMPORTANT:
%
% The test set is NOT used here.
%
% We try several thresholds and select the one producing the highest
% validation Sharpe ratio.

fprintf('Optimizing threshold using VALIDATION ONLY...\n');

nThresh = numel(thresholdGrid);

valSharpe = NaN(nThresh,1);

valReturn = NaN(nThresh,1);

valMaxDD = NaN(nThresh,1);

valTrades = NaN(nThresh,1);

for k = 1:nThresh

    th = thresholdGrid(k);

    valPos = ...
        predictionToPosition( ...
        predVal, ...
        th);

    bt = ...
        runBacktestFromReturns( ...
        yVal, ...
        valPos, ...
        initialCapital, ...
        feeRate);

    valSharpe(k) = bt.sharpe;

    valReturn(k) = bt.returnPct;

    valMaxDD(k) = bt.maxDrawdownPct;

    valTrades(k) = bt.trades;

end

%% ========================================================================
% 12. SELECT VALIDATION THRESHOLD
% =========================================================================

bestScore = -Inf;

bestK = 1;

for k = 1:nThresh

    score = valSharpe(k);

    if ~isfinite(score)

        continue;

    end

    if score > bestScore + 1e-12

        bestScore = score;

        bestK = k;

    elseif abs(score-bestScore) <= 1e-12

        % Tie-breaker:
        % higher validation return

        if valReturn(k) > valReturn(bestK)

            bestK = k;

        end

    end

end

selectedThreshold = ...
    thresholdGrid(bestK);

fprintf('Selected threshold   : %.2f%%\n', ...
    selectedThreshold*100);

fprintf('Validation Sharpe    : %.4f\n', ...
    valSharpe(bestK));

fprintf('Validation return    : %.2f%%\n', ...
    valReturn(bestK));

fprintf('Validation max DD    : %.2f%%\n', ...
    valMaxDD(bestK));

fprintf('Validation trades    : %d\n\n', ...
    valTrades(bestK));

%% ========================================================================
% 13. REFIT FINAL MODEL ON TRAIN + VALIDATION
% =========================================================================
%
% Test remains untouched until this point.

trainValIdx = ...
    [trainIdx,valIdx];

muFinal = ...
    mean(X(trainValIdx,:),1,'omitnan');

sigmaFinal = ...
    std(X(trainValIdx,:),0,1,'omitnan');

sigmaFinal( ...
    ~isfinite(sigmaFinal) | sigmaFinal < 1e-12 ...
    ) = 1;

XtrainValN = ...
    (X(trainValIdx,:) - muFinal) ./ sigmaFinal;

XtestFinalN = ...
    (X(testIdx,:) - muFinal) ./ sigmaFinal;

yTrainVal = ...
    y(trainValIdx);

modelFinal = fitrlinear( ...
    XtrainValN, ...
    yTrainVal, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda',ridgeLambda, ...
    'Solver','lbfgs');

%% ========================================================================
% 14. FINAL TEST PREDICTIONS
% =========================================================================

predTest = ...
    predict(modelFinal,XtestFinalN);

%% ========================================================================
% 15. FINAL TEST METRICS
% =========================================================================

testRMSE = ...
    rmseMetric(yTest,predTest);

testMAE = ...
    mean(abs(yTest-predTest),'omitnan');

testMSE = ...
    mean((yTest-predTest).^2,'omitnan');

testR2 = ...
    r2Metric(yTest,predTest);

testDir = ...
    directionalAccuracy(yTest,predTest);

fprintf('============================================================\n');
fprintf(' FINAL UNTOUCHED TEST PREDICTION RESULTS\n');
fprintf('============================================================\n');

fprintf('RMSE                 : %.6f\n',testRMSE);

fprintf('MAE                  : %.6f\n',testMAE);

fprintf('MSE                  : %.6f\n',testMSE);

fprintf('R2                   : %.6f\n',testR2);

fprintf('Directional accuracy : %.2f%%\n\n', ...
    testDir*100);

%% ========================================================================
% 16. FINAL TEST TRADING SIGNALS
% =========================================================================

testPos = ...
    predictionToPosition( ...
    predTest, ...
    selectedThreshold);

%% ========================================================================
% 17. FINAL AR(5) BACKTEST
% =========================================================================

btAR5 = ...
    runBacktestFromReturns( ...
    yTest, ...
    testPos, ...
    initialCapital, ...
    feeRate);

%% ========================================================================
% 18. BUY AND HOLD BENCHMARK
% =========================================================================

bnhPos = ones(size(yTest));

btBH = ...
    runBacktestFromReturns( ...
    yTest, ...
    bnhPos, ...
    initialCapital, ...
    0);

%% ========================================================================
% 19. DISPLAY FINAL RESULTS
% =========================================================================

fprintf('============================================================\n');
fprintf(' FINAL TEST BACKTEST\n');
fprintf('============================================================\n');

fprintf('AR(5) final capital : $%.2f\n', ...
    btAR5.finalCapital);

fprintf('AR(5) return        : %.2f%%\n', ...
    btAR5.returnPct);

fprintf('AR(5) Sharpe        : %.4f\n', ...
    btAR5.sharpe);

fprintf('AR(5) Sortino       : %.4f\n', ...
    btAR5.sortino);

fprintf('AR(5) max drawdown  : %.2f%%\n', ...
    btAR5.maxDrawdownPct);

fprintf('AR(5) win rate      : %.2f%%\n', ...
    btAR5.winRate);

fprintf('AR(5) profit factor : %.4f\n', ...
    btAR5.profitFactor);

fprintf('AR(5) trades        : %d\n\n', ...
    btAR5.trades);

fprintf('Buy & Hold capital  : $%.2f\n', ...
    btBH.finalCapital);

fprintf('Buy & Hold return   : %.2f%%\n', ...
    btBH.returnPct);

fprintf('Buy & Hold Sharpe   : %.4f\n', ...
    btBH.sharpe);

fprintf('Buy & Hold max DD   : %.2f%%\n', ...
    btBH.maxDrawdownPct);

fprintf('============================================================\n\n');

%% ========================================================================
% 20. SIGNAL COUNTS
% =========================================================================

nBuy = ...
    sum(testPos == 1);

nHold = ...
    sum(testPos == 0);

nSell = ...
    sum(testPos == -1);

fprintf('Final test signals:\n');

fprintf('BUY  = %d\n',nBuy);

fprintf('HOLD = %d\n',nHold);

fprintf('SELL = %d\n\n',nSell);

%% ========================================================================
% 21. SAVE ROW-BY-ROW BACKTEST RESULTS
% =========================================================================

testDates = ...
    modelDates(testIdx);

testPrices = ...
    modelPrices(testIdx);

testSentiment = ...
    modelSentiment(testIdx);

resultTable = table( ...
    testDates, ...
    testPrices, ...
    testSentiment, ...
    yTest, ...
    predTest, ...
    testPos, ...
    btAR5.turnover, ...
    btAR5.cost, ...
    btAR5.strategyReturn, ...
    btAR5.equity, ...
    btAR5.drawdown, ...
    'VariableNames', ...
    { ...
    'Date', ...
    'Close', ...
    'Sentiment', ...
    'ActualNextDayReturn', ...
    'PredictedNextDayReturn', ...
    'Position', ...
    'Turnover', ...
    'TransactionCost', ...
    'StrategyReturn', ...
    'Equity', ...
    'Drawdown' ...
    });

writetable( ...
    resultTable, ...
    fullfile( ...
    tablesDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_BACKTEST.csv'));

%% ========================================================================
% 22. SAVE SUMMARY
% =========================================================================

summaryMetric = ...
    { ...
    'InitialCapital';
    'TransactionCost';
    'LagOrder';
    'PredictionHorizon';
    'SelectedThreshold';
    'TrainObservations';
    'ValidationObservations';
    'TestObservations';
    'ValidationRMSE';
    'ValidationMAE';
    'ValidationR2';
    'ValidationDirectionalAccuracy';
    'ValidationSharpeAtSelectedThreshold';
    'TestRMSE';
    'TestMAE';
    'TestMSE';
    'TestR2';
    'TestDirectionalAccuracy';
    'AR5FinalCapital';
    'AR5ReturnPct';
    'AR5Sharpe';
    'AR5Sortino';
    'AR5MaxDrawdownPct';
    'AR5WinRate';
    'AR5ProfitFactor';
    'AR5Trades';
    'BuyHoldFinalCapital';
    'BuyHoldReturnPct';
    'BuyHoldSharpe';
    'BuyHoldMaxDrawdownPct';
    'BuySignals';
    'HoldSignals';
    'SellSignals' ...
    };

summaryValue = ...
    [ ...
    initialCapital;
    feeRate;
    lagOrder;
    predictionHorizon;
    selectedThreshold;
    nTrain;
    nVal;
    nTest;
    valRMSE;
    valMAE;
    valR2;
    valDir;
    valSharpe(bestK);
    testRMSE;
    testMAE;
    testMSE;
    testR2;
    testDir;
    btAR5.finalCapital;
    btAR5.returnPct;
    btAR5.sharpe;
    btAR5.sortino;
    btAR5.maxDrawdownPct;
    btAR5.winRate;
    btAR5.profitFactor;
    btAR5.trades;
    btBH.finalCapital;
    btBH.returnPct;
    btBH.sharpe;
    btBH.maxDrawdownPct;
    nBuy;
    nHold;
    nSell ...
    ];

summaryTable = ...
    table( ...
    summaryMetric, ...
    summaryValue, ...
    'VariableNames', ...
    {'Metric','Value'});

writetable( ...
    summaryTable, ...
    fullfile( ...
    tablesDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_SUMMARY.csv'));

%% ========================================================================
% 23. SAVE FINAL MODEL
% =========================================================================

modelInfo = struct();

modelInfo.modelType = ...
    'Ridge-regularized AR(5)';

modelInfo.lagOrder = ...
    lagOrder;

modelInfo.ridgeLambda = ...
    ridgeLambda;

modelInfo.predictionHorizon = ...
    predictionHorizon;

modelInfo.selectedThreshold = ...
    selectedThreshold;

modelInfo.transactionCost = ...
    feeRate;

modelInfo.trainDates = ...
    [ ...
    modelDates(trainIdx(1)), ...
    modelDates(trainIdx(end)) ...
    ];

modelInfo.validationDates = ...
    [ ...
    modelDates(valIdx(1)), ...
    modelDates(valIdx(end)) ...
    ];

modelInfo.testDates = ...
    [ ...
    modelDates(testIdx(1)), ...
    modelDates(testIdx(end)) ...
    ];

modelInfo.note = ...
    ['Test data was not used for model fitting or threshold ' ...
     'selection. AR warm-up rows were removed rather than filled.'];

save( ...
    fullfile( ...
    modelsDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_MODEL.mat'), ...
    'modelFinal', ...
    'modelDev', ...
    'muFinal', ...
    'sigmaFinal', ...
    'muX', ...
    'sigmaX', ...
    'ridgeLambda', ...
    'lagOrder', ...
    'selectedThreshold', ...
    'thresholdGrid', ...
    'modelInfo');

%% ========================================================================
% 24. EQUITY CURVE
% =========================================================================

fig = figure( ...
    'Color','w', ...
    'Name','Rigorous AR(5) Backtest');

plot( ...
    testDates, ...
    btAR5.equity, ...
    'LineWidth',1.6);

hold on;

plot( ...
    testDates, ...
    btBH.equity, ...
    '--', ...
    'LineWidth',1.4);

grid on;

xlabel('Date');

ylabel('Portfolio Value ($)');

title('Final Untouched Test: AR(5) vs Buy-and-Hold');

legend( ...
    {'AR(5) Strategy','Buy-and-Hold'}, ...
    'Location','best');

saveas( ...
    fig, ...
    fullfile( ...
    figDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_EQUITY.png'));

close(fig);

%% ========================================================================
% 25. ACTUAL VS PREDICTED RETURN
% =========================================================================

fig = figure( ...
    'Color','w', ...
    'Name','Prediction');

plot( ...
    testDates, ...
    yTest*100, ...
    'LineWidth',1.0);

hold on;

plot( ...
    testDates, ...
    predTest*100, ...
    'LineWidth',1.0);

yline( ...
    selectedThreshold*100, ...
    '--');

yline( ...
    -selectedThreshold*100, ...
    '--');

grid on;

xlabel('Date');

ylabel('Return (%)');

title('Actual vs Predicted Next-Day Return');

legend( ...
    {'Actual','Predicted','+Threshold','-Threshold'}, ...
    'Location','best');

saveas( ...
    fig, ...
    fullfile( ...
    figDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_PREDICTION.png'));

close(fig);

%% ========================================================================
% 26. DRAWDOWN
% =========================================================================

fig = figure( ...
    'Color','w', ...
    'Name','Drawdown');

plot( ...
    testDates, ...
    btAR5.drawdown*100, ...
    'LineWidth',1.4);

grid on;

xlabel('Date');

ylabel('Drawdown (%)');

title('AR(5) Strategy Drawdown');

saveas( ...
    fig, ...
    fullfile( ...
    figDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_DRAWDOWN.png'));

close(fig);

%% ========================================================================
% 27. BUY / SELL SIGNALS
% =========================================================================

fig = figure( ...
    'Color','w', ...
    'Name','Signals');

plot( ...
    testDates, ...
    testPrices, ...
    'LineWidth',1.1);

hold on;

buyIdx = ...
    testPos == 1;

sellIdx = ...
    testPos == -1;

plot( ...
    testDates(buyIdx), ...
    testPrices(buyIdx), ...
    '^', ...
    'MarkerSize',6);

plot( ...
    testDates(sellIdx), ...
    testPrices(sellIdx), ...
    'v', ...
    'MarkerSize',6);

grid on;

xlabel('Date');

ylabel('BTC Close');

title('AR(5) Trading Signals');

legend( ...
    {'BTC Close','BUY','SELL'}, ...
    'Location','best');

saveas( ...
    fig, ...
    fullfile( ...
    figDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_SIGNALS.png'));

close(fig);

%% ========================================================================
% 28. FINAL MESSAGE
% =========================================================================

fprintf('\nSaved files:\n');

fprintf('  %s\n', ...
    fullfile( ...
    tablesDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_BACKTEST.csv'));

fprintf('  %s\n', ...
    fullfile( ...
    tablesDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_SUMMARY.csv'));

fprintf('  %s\n', ...
    fullfile( ...
    modelsDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_MODEL.mat'));

fprintf('  %s\n', ...
    fullfile( ...
    figDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_EQUITY.png'));

fprintf('  %s\n', ...
    fullfile( ...
    figDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_PREDICTION.png'));

fprintf('  %s\n', ...
    fullfile( ...
    figDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_DRAWDOWN.png'));

fprintf('  %s\n', ...
    fullfile( ...
    figDir, ...
    'FINAL_MASTER_AR5_RIGOROUS_SIGNALS.png'));

fprintf('\n[STEP 5 RIGOROUS BACKTEST COMPLETE]\n');

fprintf(['The final test set was not used for model fitting ' ...
         'or threshold selection.\n\n']);

end


%% =========================================================================
% LOCAL FUNCTION: FIND VARIABLE
% =========================================================================

function idx = findVariable(T,candidates)

names = string(T.Properties.VariableNames);

cleanNames = ...
    lower( ...
    regexprep( ...
    names, ...
    '[^a-zA-Z0-9]', ...
    ''));

idx = [];

for i = 1:numel(candidates)

    c = ...
        lower( ...
        regexprep( ...
        string(candidates{i}), ...
        '[^a-zA-Z0-9]', ...
        ''));

    hit = ...
        find(cleanNames == c,1);

    if ~isempty(hit)

        idx = hit;

        return;

    end

end

for i = 1:numel(candidates)

    c = ...
        lower( ...
        regexprep( ...
        string(candidates{i}), ...
        '[^a-zA-Z0-9]', ...
        ''));

    hit = ...
        find(contains(cleanNames,c),1);

    if ~isempty(hit)

        idx = hit;

        return;

    end

end

end


%% =========================================================================
% LOCAL FUNCTION: NUMERIC CONVERSION
% =========================================================================

function x = toNumeric(v)

if isnumeric(v)

    x = double(v(:));

elseif islogical(v)

    x = double(v(:));

elseif iscategorical(v)

    x = str2double(string(v(:)));

elseif isstring(v) || ischar(v)

    x = str2double(string(v(:)));

elseif iscell(v)

    x = str2double(string(v(:)));

else

    try

        x = double(v(:));

    catch

        x = NaN(numel(v),1);

    end

end

end


%% =========================================================================
% LOCAL FUNCTION: DATE PARSER
% =========================================================================

function d = parseDateColumn(v)

% ---------------------------------------------------------
% Already datetime
% ---------------------------------------------------------

if isdatetime(v)

    d = dateshift( ...
        v(:), ...
        'start', ...
        'day');

    return;

end

% ---------------------------------------------------------
% Numeric values
% ---------------------------------------------------------

if isnumeric(v)

    z = double(v(:));

    d = NaT(size(z));

    finite = isfinite(z);

    if any(finite)

        med = median(abs(z(finite)));

        try

            if med > 1e17

                % Nanoseconds

                d(finite) = ...
                    datetime( ...
                    z(finite)/1e9, ...
                    'ConvertFrom','posixtime');

            elseif med > 1e14

                % Microseconds

                d(finite) = ...
                    datetime( ...
                    z(finite)/1e6, ...
                    'ConvertFrom','posixtime');

            elseif med > 1e11

                % Milliseconds

                d(finite) = ...
                    datetime( ...
                    z(finite)/1e3, ...
                    'ConvertFrom','posixtime');

            elseif med > 1e9

                % Seconds

                d(finite) = ...
                    datetime( ...
                    z(finite), ...
                    'ConvertFrom','posixtime');

            else

                % MATLAB serial date number

                d(finite) = ...
                    datetime( ...
                    z(finite), ...
                    'ConvertFrom','datenum');

            end

        catch

            d(:) = NaT;

        end

    end

    d = dateshift( ...
        d, ...
        'start', ...
        'day');

    return;

end

% ---------------------------------------------------------
% Strings
% ---------------------------------------------------------

s = string(v(:));

d = NaT(size(s));

% Try numeric strings.

num = str2double(s);

numMask = isfinite(num);

if any(numMask)

    med = median(abs(num(numMask)));

    try

        if med > 1e17

            d(numMask) = ...
                datetime( ...
                num(numMask)/1e9, ...
                'ConvertFrom','posixtime');

        elseif med > 1e14

            d(numMask) = ...
                datetime( ...
                num(numMask)/1e6, ...
                'ConvertFrom','posixtime');

        elseif med > 1e11

            d(numMask) = ...
                datetime( ...
                num(numMask)/1e3, ...
                'ConvertFrom','posixtime');

        elseif med > 1e9

            d(numMask) = ...
                datetime( ...
                num(numMask), ...
                'ConvertFrom','posixtime');

        elseif med > 700000

            d(numMask) = ...
                datetime( ...
                num(numMask), ...
                'ConvertFrom','datenum');

        end

    catch

    end

end

% ---------------------------------------------------------
% Standard date formats
% ---------------------------------------------------------

formats = ...
    { ...
    'yyyy-MM-dd HH:mm:ss.SSSSSS', ...
    'yyyy-MM-dd HH:mm:ss.SSS', ...
    'yyyy-MM-dd HH:mm:ss', ...
    'yyyy-MM-dd HH:mm', ...
    'yyyy-MM-dd', ...
    'yyyy/MM/dd HH:mm:ss', ...
    'yyyy/MM/dd HH:mm', ...
    'yyyy/MM/dd', ...
    'dd-MM-yyyy HH:mm:ss', ...
    'dd-MM-yyyy HH:mm', ...
    'dd-MM-yyyy', ...
    'dd/MM/yyyy HH:mm:ss', ...
    'dd/MM/yyyy HH:mm', ...
    'dd/MM/yyyy' ...
    };

remaining = isnat(d);

for k = 1:numel(formats)

    if ~any(remaining)

        break;

    end

    try

        tmp = ...
            datetime( ...
            s(remaining), ...
            'InputFormat', ...
            formats{k});

        d(remaining) = tmp;

    catch

    end

    remaining = isnat(d);

end

% ---------------------------------------------------------
% Final flexible MATLAB parser
% ---------------------------------------------------------

remaining = isnat(d);

if any(remaining)

    try

        d(remaining) = ...
            datetime(s(remaining));

    catch

    end

end

d = dateshift( ...
    d, ...
    'start', ...
    'day');

end


%% =========================================================================
% LOCAL FUNCTION: CONVERT PREDICTION TO POSITION
% =========================================================================

function p = predictionToPosition(pred,threshold)

p = zeros(size(pred));

% BUY
p(pred >= threshold) = 1;

% SELL
p(pred <= -threshold) = -1;

% Everything between the two thresholds stays HOLD.

end


%% =========================================================================
% LOCAL FUNCTION: BACKTEST
% =========================================================================

function bt = runBacktestFromReturns( ...
    actualReturn, ...
    position, ...
    initialCapital, ...
    feeRate)

actualReturn = actualReturn(:);

position = position(:);

if numel(actualReturn) ~= numel(position)

    error( ...
        'Backtest return/position lengths do not match.');

end

n = numel(actualReturn);

%% ---------------------------------------------------------
% POSITION TURNOVER
% ----------------------------------------------------------

turnover = zeros(n,1);

previousPosition = 0;

for t = 1:n

    turnover(t) = ...
        abs(position(t)-previousPosition);

    previousPosition = position(t);

end

%% ---------------------------------------------------------
% TRANSACTION COST
% ----------------------------------------------------------

cost = ...
    feeRate .* turnover;

%% ---------------------------------------------------------
% STRATEGY RETURN
% ----------------------------------------------------------

strategyReturn = ...
    position .* actualReturn - cost;

%% ---------------------------------------------------------
% EQUITY CURVE
% ----------------------------------------------------------

equity = zeros(n,1);

equity(1) = ...
    initialCapital * ...
    (1 + strategyReturn(1));

for t = 2:n

    equity(t) = ...
        equity(t-1) * ...
        (1 + strategyReturn(t));

end

%% ---------------------------------------------------------
% DRAWDOWN
% ----------------------------------------------------------

peak = cummax(equity);

drawdown = ...
    equity ./ peak - 1;

%% ---------------------------------------------------------
% SHARPE RATIO
% ----------------------------------------------------------

dailyMean = ...
    mean(strategyReturn,'omitnan');

dailyStd = ...
    std(strategyReturn,0,'omitnan');

if isfinite(dailyStd) && dailyStd > 0

    sharpe = ...
        sqrt(365) * ...
        dailyMean / ...
        dailyStd;

else

    sharpe = NaN;

end

%% ---------------------------------------------------------
% SORTINO RATIO
% ----------------------------------------------------------

downside = ...
    min(strategyReturn,0);

downsideStd = ...
    std(downside,0,'omitnan');

if isfinite(downsideStd) && downsideStd > 0

    sortino = ...
        sqrt(365) * ...
        dailyMean / ...
        downsideStd;

else

    sortino = NaN;

end

%% ---------------------------------------------------------
% PROFIT FACTOR
% ----------------------------------------------------------

grossProfit = ...
    sum( ...
    strategyReturn(strategyReturn > 0), ...
    'omitnan');

grossLoss = ...
    -sum( ...
    strategyReturn(strategyReturn < 0), ...
    'omitnan');

if grossLoss > 0

    profitFactor = ...
        grossProfit / grossLoss;

else

    profitFactor = Inf;

end

%% ---------------------------------------------------------
% TRADE COUNT
% ---------------------------------------------------------

trades = ...
    sum(turnover > 0);

%% ---------------------------------------------------------
% WIN RATE
% ---------------------------------------------------------

active = ...
    position ~= 0;

if any(active)

    winRate = ...
        mean( ...
        strategyReturn(active) > 0, ...
        'omitnan');

else

    winRate = NaN;

end

%% ---------------------------------------------------------
% OUTPUT
% ----------------------------------------------------------

bt = struct();

bt.turnover = turnover;

bt.cost = cost;

bt.strategyReturn = strategyReturn;

bt.equity = equity;

bt.drawdown = drawdown;

bt.finalCapital = ...
    equity(end);

bt.returnPct = ...
    (equity(end)/initialCapital - 1) * 100;

bt.sharpe = ...
    sharpe;

bt.sortino = ...
    sortino;

bt.maxDrawdownPct = ...
    min(drawdown) * 100;

bt.winRate = ...
    winRate * 100;

bt.profitFactor = ...
    profitFactor;

bt.trades = ...
    trades;

end


%% =========================================================================
% LOCAL FUNCTION: RMSE
% =========================================================================

function value = rmseMetric(y,yp)

e = ...
    y(:) - yp(:);

value = ...
    sqrt( ...
    mean( ...
    e.^2, ...
    'omitnan'));

end


%% =========================================================================
% LOCAL FUNCTION: R2
% =========================================================================

function value = r2Metric(y,yp)

y = y(:);

yp = yp(:);

valid = ...
    isfinite(y) & ...
    isfinite(yp);

y = y(valid);

yp = yp(valid);

den = ...
    sum( ...
    (y-mean(y)).^2);

if den <= eps

    value = NaN;

else

    value = ...
        1 - ...
        sum((y-yp).^2) / ...
        den;

end

end


%% =========================================================================
% LOCAL FUNCTION: DIRECTIONAL ACCURACY
% =========================================================================

function value = directionalAccuracy(actual,predicted)

actual = actual(:);

predicted = predicted(:);

valid = ...
    isfinite(actual) & ...
    isfinite(predicted);

actual = actual(valid);

predicted = predicted(valid);

actualSign = ...
    sign(actual);

predSign = ...
    sign(predicted);

value = ...
    mean(actualSign == predSign);

end