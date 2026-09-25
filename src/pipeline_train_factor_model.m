%% ============================================================
% PIPELINE: TRAIN FACTOR MODEL
% ============================================================
% Step B of the project plan - corresponds to the brief's Step 5
% (time series/factor model with sentiment as a factor) and Step 6
% (trading strategy + backtest).
%
% Approach: a linear regression predicting next-day return, with
% STEPWISE FACTOR SELECTION (stepwiselm) rather than throwing every
% factor in and reporting a single accuracy number. This directly
% answers "which factors/parameters actually matter" - the selected
% coefficients, standard errors, and p-values are saved and plotted,
% not just implied by a black-box model.
%
% Why this over deep learning / reinforcement learning: the usable
% dataset here is only the ~1,186-day overlap between the price and
% sentiment sources (see pipeline_build_dataset.m). That's thin for
% approaches that typically need many thousands of samples to
% generalize, and - since I can't execute MATLAB in the environment
% used to write this - a small interpretable model is also the one
% where "should be correct" and "is correct" are closest together
% without a training run to check.
%
% IMPORTANT: this has not been executed (no MATLAB/Octave available
% where it was written). Run it and read the printed factor table
% before trusting any result - see README.md for what to check.
%
% Produces:
%   results/factor_model.mat              (fitted model + normalization)
%   results/factor_model_coefficients.csv (every selected factor: estimate,
%                                           SE, t-stat, p-value)
%   results/factor_model_results.mat      (full results struct)
%   results/predictions.csv               (test-period predictions + signals)
%   results/factor_significance.png
%   results/actual_vs_predicted.png
%   results/equity_curve.png
%   results/drawdown.png
%   results/signals.png
% ============================================================

function pipeline_train_factor_model()
%PIPELINE_TRAIN_FACTOR_MODEL Legacy factor model pipeline script.

clc;
close all;

fprintf('\n============================================================\n');
fprintf(' PIPELINE: TRAIN FACTOR MODEL\n');
fprintf('============================================================\n\n');

thisFile = mfilename('fullpath');
if isempty(thisFile)
    srcDir = pwd;
else
    srcDir = fileparts(thisFile);
end
projectDir = fileparts(srcDir);

addpath(srcDir);
lib = factorTradingLib();

processedFile = fullfile(projectDir, 'data', 'processed', 'merged_price_sentiment.csv');
resultsDir = fullfile(projectDir, 'results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

if ~isfile(processedFile)
    error('pipeline_train_factor_model:missingData', ...
        'Run pipeline_build_dataset.m first to generate %s', processedFile);
end

cfg = lib.config();

%% --- Load merged data ---
merged = readtable(processedFile);
fprintf('Loaded %d days of merged price+sentiment data.\n', height(merged));
fprintf('Date range: %s -> %s\n', ...
    datestr(merged.Date(1), 'dd-mmm-yyyy'), datestr(merged.Date(end), 'dd-mmm-yyyy'));

%% --- Factor engineering ---
fprintf('\n============================================================\n');
fprintf('                 FACTOR ENGINEERING\n');
fprintf('============================================================\n\n');

[X, factorNames] = lib.engineerFactors(merged);
[y, validTarget] = lib.createReturnTarget(merged.Close, cfg.predictionHorizon);

fprintf('Factors engineered: %d\n', numel(factorNames));
for i = 1:numel(factorNames)
    fprintf('  %2d. %s\n', i, factorNames{i});
end

%% --- Remove invalid rows (feature warm-up period + target edge) ---
valid = validTarget & all(isfinite(X), 2) & isfinite(merged.Close);

X = X(valid, :);
y = y(valid);
dates = merged.Date(valid);
price = merged.Close(valid);

fprintf('\nValid rows after removing warm-up/edge NaNs: %d\n', length(y));

%% --- Chronological split ---
N = length(y);
[trainIdx, validationIdx, testIdx] = lib.trainValTestSplit(N, cfg.trainRatio, cfg.validationRatio);

fprintf('\n============================================================\n');
fprintf('                     DATA SPLIT\n');
fprintf('============================================================\n\n');
fprintf('Training samples   : %d (%s -> %s)\n', length(trainIdx), ...
    datestr(dates(trainIdx(1)), 'dd-mmm-yyyy'), datestr(dates(trainIdx(end)), 'dd-mmm-yyyy'));
fprintf('Validation samples : %d (%s -> %s)\n', length(validationIdx), ...
    datestr(dates(validationIdx(1)), 'dd-mmm-yyyy'), datestr(dates(validationIdx(end)), 'dd-mmm-yyyy'));
fprintf('Testing samples    : %d (%s -> %s)\n', length(testIdx), ...
    datestr(dates(testIdx(1)), 'dd-mmm-yyyy'), datestr(dates(testIdx(end)), 'dd-mmm-yyyy'));

if length(trainIdx) < 150
    warning('pipeline_train_factor_model:smallTrainingSet', ...
        ['Only %d training samples. With %d candidate factors, stepwise ' ...
         'selection on a set this size can still overfit - treat the ' ...
         'selected factor list as a hypothesis to check, not a settled ' ...
         'result.'], length(trainIdx), numel(factorNames));
end

%% --- Training-only normalization ---
mu = mean(X(trainIdx,:), 1, 'omitnan');
sigma = std(X(trainIdx,:), 0, 1, 'omitnan');
sigma(sigma < 1e-12) = 1;
Xnorm = (X - mu) ./ sigma;

%% --- Fit factor model with stepwise selection (TRAIN split only) ---
fprintf('\n============================================================\n');
fprintf('        FITTING FACTOR MODEL (stepwise selection)\n');
fprintf('============================================================\n\n');

modelInfo = lib.fitFactorModel(Xnorm(trainIdx,:), y(trainIdx), factorNames, cfg);

fprintf('Selected %d of %d candidate factors:\n\n', ...
    numel(modelInfo.selectedNames), numel(factorNames));
disp(modelInfo.coefficients);
fprintf('In-sample R^2 (train)     : %.4f\n', modelInfo.rSquared);
fprintf('In-sample Adjusted R^2    : %.4f\n', modelInfo.adjRSquared);

writetable(cell2table([modelInfo.coefficients.Properties.RowNames, ...
    num2cell(modelInfo.coefficients.Variables)], ...
    'VariableNames', [{'Factor'}, modelInfo.coefficients.Properties.VariableNames]), ...
    fullfile(resultsDir, 'factor_model_coefficients.csv'));

lib.plotFactorSignificance(modelInfo.coefficients, ...
    'Selected Factors: Coefficient and Significance', ...
    fullfile(resultsDir, 'factor_significance.png'));

%% --- Validation performance (informational) ---
fprintf('\n============================================================\n');
fprintf('              VALIDATION EVALUATION\n');
fprintf('============================================================\n\n');

valPredicted = lib.predictReturns(modelInfo, Xnorm(validationIdx,:));
valReport = lib.regressionReport(y(validationIdx), valPredicted);

fprintf('Validation RMSE                : %.6f\n', valReport.rmse);
fprintf('Validation MAE                 : %.6f\n', valReport.mae);
fprintf('Validation R^2 (out-of-sample) : %.4f\n', valReport.rSquared);
fprintf('Validation Directional Accuracy: %.2f %%\n', 100*valReport.directionalAccuracy);

%% --- Final untouched test evaluation ---
fprintf('\n============================================================\n');
fprintf('           FINAL UNTOUCHED TEST EVALUATION\n');
fprintf('============================================================\n\n');

testPredicted = lib.predictReturns(modelInfo, Xnorm(testIdx,:));
testReport = lib.regressionReport(y(testIdx), testPredicted);

fprintf('Test RMSE                : %.6f\n', testReport.rmse);
fprintf('Test MAE                 : %.6f\n', testReport.mae);
fprintf('Test R^2 (out-of-sample) : %.4f\n', testReport.rSquared);
fprintf('Test Directional Accuracy: %.2f %%\n', 100*testReport.directionalAccuracy);

lib.plotActualVsPredicted(y(testIdx), testPredicted, ...
    'Test Period: Actual vs Predicted Next-Day Return', ...
    fullfile(resultsDir, 'actual_vs_predicted.png'));

%% --- Convert predictions to trading signals + backtest ---
fprintf('\n============================================================\n');
fprintf('                 TRADING SIGNALS + BACKTEST\n');
fprintf('============================================================\n\n');

signals = lib.returnsToSignal(testPredicted, cfg.buyThreshold, cfg.sellThreshold);

buySignals  = sum(strcmp(signals, 'BUY'));
holdSignals = sum(strcmp(signals, 'HOLD'));
sellSignals = sum(strcmp(signals, 'SELL'));
fprintf('Signal distribution: BUY=%d, HOLD=%d, SELL=%d\n', buySignals, holdSignals, sellSignals);

testPrices = price(testIdx);
bt = lib.runBacktest(testPrices, signals, cfg.initialCapital);

fprintf('\nInitial Capital       : $%.2f\n', cfg.initialCapital);
fprintf('Final Strategy Value  : $%.2f\n', bt.capital);
fprintf('Strategy Return       : %.2f %%\n', bt.strategyReturnPercent);
fprintf('Buy & Hold Return     : %.2f %%\n', bt.buyHoldReturnPercent);
fprintf('Sharpe Ratio          : %.4f\n', bt.sharpeRatio);
fprintf('Maximum Drawdown      : %.2f %%\n', bt.maximumDrawdownPercent);
fprintf('Number of Trades      : %d\n', bt.tradeCount);
fprintf('Win Rate              : %.2f %%\n', bt.winRatePercent);

lib.plotEquityCurve(bt.equity, 'Test Period Equity Curve', ...
    fullfile(resultsDir, 'equity_curve.png'));
lib.plotDrawdown(bt.drawdown, 'Test Period Drawdown', ...
    fullfile(resultsDir, 'drawdown.png'));
lib.plotSignalChart(testPrices, signals, 'Test Period Trading Signals', ...
    fullfile(resultsDir, 'signals.png'));

%% --- Save predictions ---
predictionTable = table(dates(testIdx), testPrices, y(testIdx), testPredicted, signals, ...
    'VariableNames', {'Date','Close','ActualReturn','PredictedReturn','Signal'});
writetable(predictionTable, fullfile(resultsDir, 'predictions.csv'));

%% --- Save results struct ---
results = struct();
results.Model = 'Sentiment Factor Model (stepwise linear regression)';
results.PredictionHorizon = cfg.predictionHorizon;
results.SelectedFactors = modelInfo.selectedNames;
results.TrainingSamples = length(trainIdx);
results.ValidationSamples = length(validationIdx);
results.TestingSamples = length(testIdx);
results.InSampleRSquared = modelInfo.rSquared;
results.InSampleAdjRSquared = modelInfo.adjRSquared;
results.ValidationRMSE = valReport.rmse;
results.ValidationRSquared = valReport.rSquared;
results.ValidationDirectionalAccuracy = 100*valReport.directionalAccuracy;
results.TestRMSE = testReport.rmse;
results.TestRSquared = testReport.rSquared;
results.TestDirectionalAccuracy = 100*testReport.directionalAccuracy;
results.InitialCapital = cfg.initialCapital;
results.FinalStrategyValue = bt.capital;
results.StrategyReturn = bt.strategyReturnPercent;
results.BuyHoldReturn = bt.buyHoldReturnPercent;
results.SharpeRatio = bt.sharpeRatio;
results.MaximumDrawdown = bt.maximumDrawdownPercent;
results.NumberOfTrades = bt.tradeCount;
results.WinRate = bt.winRatePercent;
results.BuySignals = buySignals;
results.HoldSignals = holdSignals;
results.SellSignals = sellSignals;
results.StartDate = dates(testIdx(1));
results.EndDate = dates(testIdx(end));

save(fullfile(resultsDir, 'factor_model.mat'), 'modelInfo', 'mu', 'sigma', 'factorNames', 'cfg');
save(fullfile(resultsDir, 'factor_model_results.mat'), 'results');

%% --- Summary ---
fprintf('\n============================================================\n');
fprintf('       PIPELINE COMPLETED SUCCESSFULLY\n');
fprintf('============================================================\n\n');
fprintf('Selected factors (%d): %s\n', numel(results.SelectedFactors), ...
    strjoin(results.SelectedFactors, ', '));
fprintf('\nTEST PERIOD\n');
fprintf('Directional Accuracy : %.2f %%\n', results.TestDirectionalAccuracy);
fprintf('R^2                  : %.4f\n', results.TestRSquared);
fprintf('Strategy Return      : %.2f %%\n', results.StrategyReturn);
fprintf('Buy & Hold           : %.2f %%\n', results.BuyHoldReturn);
fprintf('Sharpe Ratio         : %.4f\n', results.SharpeRatio);
fprintf('Maximum Drawdown     : %.2f %%\n', results.MaximumDrawdown);
fprintf('\nGenerated files (in results/):\n');
fprintf('factor_model.mat, factor_model_results.mat, factor_model_coefficients.csv\n');
fprintf('predictions.csv, factor_significance.png, actual_vs_predicted.png\n');
fprintf('equity_curve.png, drawdown.png, signals.png\n');
fprintf('\n============================================================\n');

end
