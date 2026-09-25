function p = plotUtils()
%PLOTUTILS Clean, publication-quality visualization routines.
%
%   P = PLOTUTILS() returns a struct of function handles.

p.plotEDA                  = @plotEDA;
p.plotSentimentComparison  = @plotSentimentComparison;
p.plotFactorSignificance   = @plotFactorSignificance;
p.plotModelComparison      = @plotModelComparison;
p.plotEquityCurves         = @plotEquityCurves;
p.plotDrawdowns            = @plotDrawdowns;
p.plotSignalsOnPrice       = @plotSignalsOnPrice;

end

%% ------------------------------------------------------------
function plotEDA(dates, price, sentiment, savePath)
f = figure('Position', [100, 100, 1000, 700], 'Visible', 'off');

subplot(3, 1, 1);
plot(dates, price, 'LineWidth', 1.5, 'Color', [0.15 0.35 0.75]);
grid on;
title('Bitcoin Daily Close Price (USD)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Price ($)');

subplot(3, 1, 2);
plot(dates, sentiment, 'LineWidth', 1.2, 'Color', [0.85 0.45 0.1]);
hold on;
yline(mean(sentiment, 'omitnan'), '--r', 'Mean Sentiment');
grid on;
title('Daily Consolidated Sentiment Score', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Sentiment Score');

subplot(3, 1, 3);
returns = [0; diff(price) ./ price(1:end-1)];
rollingCorr = zeros(length(returns), 1);
w = 30;
for i = w:length(returns)
    r_sub = returns(i-w+1:i);
    s_sub = sentiment(i-w+1:i);
    c = corrcoef(r_sub, s_sub);
    rollingCorr(i) = c(1, 2);
end
plot(dates, rollingCorr, 'LineWidth', 1.2, 'Color', [0.2 0.65 0.3]);
yline(0, ':k');
grid on;
title('30-Day Rolling Correlation: BTC Returns vs Sentiment', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Pearson r');
xlabel('Date');

if nargin >= 4 && ~isempty(savePath)
    saveas(f, savePath);
    close(f);
end
end

%% ------------------------------------------------------------
function plotSentimentComparison(scores, vaderLabels, ratioLabels, savePath)
f = figure('Position', [100, 100, 900, 450], 'Visible', 'off');

subplot(1, 2, 1);
histogram(categorical(vaderLabels), 'FaceColor', [0.2 0.4 0.8]);
title('VADER-Style Classification Distribution', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Number of Days');
grid on;

subplot(1, 2, 2);
histogram(categorical(ratioLabels), 'FaceColor', [0.8 0.3 0.2]);
title('Ratio-Rule Classification Distribution', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Number of Days');
grid on;

if nargin >= 4 && ~isempty(savePath)
    saveas(f, savePath);
    close(f);
end
end

%% ------------------------------------------------------------
function plotFactorSignificance(coefficients, figureTitle, savePath)
f = figure('Position', [100, 100, 850, 500], 'Visible', 'off');

names = coefficients.Properties.RowNames;
keep = ~strcmp(names, '(Intercept)');
names = names(keep);

if isempty(names)
    text(0.5, 0.5, 'No individual factors passed significance entry threshold (p < 0.05)', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title(figureTitle, 'FontSize', 12, 'FontWeight', 'bold');
else
    estimates = coefficients.Estimate(keep);
    pVals = coefficients.pValue(keep);
    
    bar(estimates, 'FaceColor', [0.25 0.5 0.75]);
    set(gca, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 45);
    ylabel('Regression Coefficient');
    title(figureTitle, 'FontSize', 12, 'FontWeight', 'bold');
    grid on;
    
    for i = 1:numel(names)
        text(i, estimates(i), sprintf('p=%.3f', pVals(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9);
    end
end

if nargin >= 3 && ~isempty(savePath)
    saveas(f, savePath);
    close(f);
end
end

%% ------------------------------------------------------------
function plotModelComparison(modelNames, rmseVals, daVals, savePath)
f = figure('Position', [100, 100, 900, 450], 'Visible', 'off');

subplot(1, 2, 1);
bar(rmseVals * 100, 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'XTick', 1:numel(modelNames), 'XTickLabel', modelNames, 'XTickLabelRotation', 25);
ylabel('RMSE (%)');
title('Out-of-Sample RMSE (Lower is Better)', 'FontSize', 11, 'FontWeight', 'bold');
grid on;

subplot(1, 2, 2);
bar(daVals, 'FaceColor', [0.2 0.6 0.4]);
set(gca, 'XTick', 1:numel(modelNames), 'XTickLabel', modelNames, 'XTickLabelRotation', 25);
ylabel('Directional Accuracy (%)');
yline(50, '--r', 'Random Guess (50%)');
title('Directional Accuracy (Higher is Better)', 'FontSize', 11, 'FontWeight', 'bold');
grid on;

if nargin >= 4 && ~isempty(savePath)
    saveas(f, savePath);
    close(f);
end
end

%% ------------------------------------------------------------
function plotEquityCurves(dates, equityMap, savePath)
f = figure('Position', [100, 100, 950, 550], 'Visible', 'off');

colors = lines(numel(fieldnames(equityMap)));
names = fieldnames(equityMap);
hold on;

for k = 1:numel(names)
    plot(dates, equityMap.(names{k}), 'LineWidth', 1.8, 'Color', colors(k, :), 'DisplayName', names{k});
end

grid on;
legend('Location', 'northwest', 'FontSize', 10);
title('Portfolio Equity Curves Comparison (Out-of-Sample Test Period)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Date');
ylabel('Portfolio Capital ($)');
hold off;

if nargin >= 3 && ~isempty(savePath)
    saveas(f, savePath);
    close(f);
end
end

%% ------------------------------------------------------------
function plotDrawdowns(dates, drawdownMap, savePath)
f = figure('Position', [100, 100, 950, 500], 'Visible', 'off');

colors = lines(numel(fieldnames(drawdownMap)));
names = fieldnames(drawdownMap);
hold on;

for k = 1:numel(names)
    plot(dates, drawdownMap.(names{k}) * 100, 'LineWidth', 1.5, 'Color', colors(k, :), 'DisplayName', names{k});
end

grid on;
legend('Location', 'southwest', 'FontSize', 10);
title('Underwater Drawdown Profiles', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Date');
ylabel('Drawdown (%)');
hold off;

if nargin >= 3 && ~isempty(savePath)
    saveas(f, savePath);
    close(f);
end
end

%% ------------------------------------------------------------
function plotSignalsOnPrice(dates, prices, signals, savePath)
f = figure('Position', [100, 100, 1000, 550], 'Visible', 'off');

plot(dates, prices, 'LineWidth', 1.2, 'Color', [0.3 0.3 0.3], 'DisplayName', 'BTC Price');
hold on;

buyIdx  = find(strcmp(signals, 'BUY'));
sellIdx = find(strcmp(signals, 'SELL'));

if ~isempty(buyIdx)
    scatter(dates(buyIdx), prices(buyIdx), 50, 'g', '^', 'filled', 'DisplayName', 'BUY Signal');
end
if ~isempty(sellIdx)
    scatter(dates(sellIdx), prices(sellIdx), 50, 'r', 'v', 'filled', 'DisplayName', 'SELL Signal');
end

grid on;
legend('Location', 'best', 'FontSize', 10);
title('Trading Signals Generated on Out-of-Sample Test Period', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Date');
ylabel('Price ($)');
hold off;

if nargin >= 4 && ~isempty(savePath)
    saveas(f, savePath);
    close(f);
end
end
