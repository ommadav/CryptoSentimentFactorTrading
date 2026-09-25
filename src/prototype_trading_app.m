function prototype_trading_app
% ==============================================================
% CRYPTO SENTIMENT TRADING SYSTEM
% MATLAB INTERACTIVE PROTOTYPE DASHBOARD
%
% FINAL WORKING VERSION
%
% MODEL:
%       AR(5) Ridge Regression
%
% TRAINING:
%       Chronological historical BTC data
%       Train-only standardization
%       Lambda = 1e-4
%       Final refit on Train + Validation
%
% SIGNAL:
%       Step 10B threshold = 0.00086042
%
% IMPORTANT:
%       Sentiment and RSI are contextual information only.
%       They are NOT injected into the AR(5) prediction.
%
%       Signal Strength is NOT a probability.
%
%       This is an academic historical research prototype.
%       It is NOT a live trading system.
%
% ==============================================================

clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       CRYPTO SENTIMENT TRADING SYSTEM\n');
fprintf('             MATLAB PROTOTYPE DASHBOARD\n');
fprintf('============================================================\n\n');

%% =============================================================
% PROJECT PATHS
% =============================================================

thisFile = mfilename('fullpath');
srcFolder = fileparts(thisFile);
projectRoot = fileparts(srcFolder);

btcFile = fullfile( ...
    projectRoot, ...
    'data', ...
    'raw', ...
    'btc_1d_data_2018_to_2025.csv');

sentimentFile = fullfile( ...
    projectRoot, ...
    'data', ...
    'raw', ...
    'sentiment_data_consolidated.csv');

modelFile = fullfile( ...
    projectRoot, ...
    'results', ...
    'models', ...
    'FINAL_MASTER_AR5_MODEL.mat');

%% =============================================================
% CHECK FILES
% =============================================================

fprintf('Checking project files...\n');

if isfile(btcFile)
    fprintf('  [OK] btc_1d_data_2018_to_2025.csv\n');
else
    error('BTC CSV not found:\n%s',btcFile);
end

if isfile(sentimentFile)
    fprintf('  [OK] sentiment_data_consolidated.csv\n');
else
    error('Sentiment CSV not found:\n%s',sentimentFile);
end

if isfile(modelFile)
    fprintf('  [OK] FINAL_MASTER_AR5_MODEL.mat\n');
else
    fprintf('  [INFO] Model MAT file not used for interactive prediction.\n');
end

%% =============================================================
% LOAD BTC
% =============================================================

fprintf('\nLoading BTC data...\n');

[btcDate,...
 btcOpen,...
 btcHigh,...
 btcLow,...
 btcClose,...
 btcVolume] = readBTCraw(btcFile);

%% =============================================================
% CLEAN BTC
% =============================================================

valid = ...
    ~isnat(btcDate) & ...
    isfinite(btcClose) & ...
    btcClose > 0;

btcDate   = btcDate(valid);
btcOpen   = btcOpen(valid);
btcHigh   = btcHigh(valid);
btcLow    = btcLow(valid);
btcClose  = btcClose(valid);
btcVolume = btcVolume(valid);

%% =============================================================
% SORT
% =============================================================

[btcDate,idx] = sort(btcDate);

btcOpen   = btcOpen(idx);
btcHigh   = btcHigh(idx);
btcLow    = btcLow(idx);
btcClose  = btcClose(idx);
btcVolume = btcVolume(idx);

%% =============================================================
% REMOVE DUPLICATE DATES
% =============================================================

[btcDate,idx] = unique(btcDate,'last');

btcOpen   = btcOpen(idx);
btcHigh   = btcHigh(idx);
btcLow    = btcLow(idx);
btcClose  = btcClose(idx);
btcVolume = btcVolume(idx);

fprintf('\nBTC DATE STATUS:\n');

fprintf( ...
    '  Valid dates : %d\n', ...
    numel(btcDate));

fprintf( ...
    '  First date  : %s\n', ...
    datestr(btcDate(1),'dd-mmm-yyyy'));

fprintf( ...
    '  Last date   : %s\n', ...
    datestr(btcDate(end),'dd-mmm-yyyy'));

fprintf( ...
    'BTC rows after cleaning: %d\n', ...
    numel(btcClose));

%% =============================================================
% LOAD SENTIMENT
% =============================================================

fprintf('\nLoading sentiment data...\n');

sentimentTable = readtable( ...
    sentimentFile, ...
    'VariableNamingRule','preserve');

fprintf( ...
    'Sentiment rows: %d\n', ...
    height(sentimentTable));

sentimentDateColumn = findColumn( ...
    sentimentTable, ...
    {'time','Time','Date','date','timestamp','Timestamp'});

sentimentColumn = findColumn( ...
    sentimentTable, ...
    {'sentiment','Sentiment'});

if isempty(sentimentDateColumn)
    error('Could not locate sentiment date column.');
end

if isempty(sentimentColumn)
    error('Could not locate sentiment column.');
end

sentimentDates = convertDates( ...
    sentimentTable.( ...
    sentimentTable.Properties.VariableNames{ ...
    sentimentDateColumn}));

sentimentValues = convertNumbers( ...
    sentimentTable.( ...
    sentimentTable.Properties.VariableNames{ ...
    sentimentColumn}));

validSentiment = ...
    ~isnat(sentimentDates) & ...
    isfinite(sentimentValues);

sentimentDates = sentimentDates(validSentiment);
sentimentValues = sentimentValues(validSentiment);

[sentimentDates,idx] = sort(sentimentDates);

sentimentValues = sentimentValues(idx);

%% =============================================================
% COMMON DATE
% =============================================================

latestBTCDate = max(btcDate);

latestSentimentDate = max(sentimentDates);

commonDate = min( ...
    latestBTCDate, ...
    latestSentimentDate);

fprintf('\n');
fprintf('============================================================\n');
fprintf('COMMON DATA DATE\n');
fprintf('============================================================\n');

fprintf( ...
    'Latest BTC date       : %s\n', ...
    datestr(latestBTCDate,'dd-mmm-yyyy'));

fprintf( ...
    'Latest sentiment date : %s\n', ...
    datestr(latestSentimentDate,'dd-mmm-yyyy'));

fprintf( ...
    'Dashboard date        : %s\n', ...
    datestr(commonDate,'dd-mmm-yyyy'));

fprintf('============================================================\n');

%% =============================================================
% FIND COMMON BTC ROW
% =============================================================

btcDays = dateshift(btcDate,'start','day');

commonDay = dateshift(commonDate,'start','day');

btcIndex = find( ...
    btcDays == commonDay, ...
    1, ...
    'last');

if isempty(btcIndex)

    btcIndex = find( ...
        btcDate <= commonDate, ...
        1, ...
        'last');

end

if isempty(btcIndex)

    error('Could not locate BTC data for common date.');

end

%% =============================================================
% BUILD AR(5) MODEL
%
% This avoids depending on unknown variable names inside the
% saved MAT file.
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('BUILDING AR(5) RIDGE MODEL\n');
fprintf('============================================================\n');

ar5Model = buildAR5Model( ...
    btcDate, ...
    btcClose);

fprintf( ...
    'AR(5) Lambda          : %.6f\n', ...
    ar5Model.lambda);

fprintf( ...
    'Training observations : %d\n', ...
    numel(ar5Model.trainY));

fprintf( ...
    'Validation observations: %d\n', ...
    numel(ar5Model.valY));

fprintf( ...
    'Final refit observations: %d\n', ...
    numel(ar5Model.refitY));

fprintf( ...
    'Validation RMSE       : %.6f\n', ...
    ar5Model.validationRMSE);

fprintf( ...
    'Validation MAE        : %.6f\n', ...
    ar5Model.validationMAE);

fprintf('============================================================\n');

%% =============================================================
% DEFAULT MARKET VALUES
% =============================================================

currentPrice = btcClose(btcIndex);

if btcIndex > 1

    previousClose = btcClose(btcIndex-1);

else

    previousClose = currentPrice;

end

currentVolume = btcVolume(btcIndex);

latestSentiment = getSentimentAtDate( ...
    sentimentDates, ...
    sentimentValues, ...
    commonDate);

%% =============================================================
% RSI
% =============================================================

priceHistory = btcClose(1:btcIndex);

rsiValues = calculateRSI( ...
    priceHistory, ...
    14);

latestRSI = rsiValues(end);

if ~isfinite(latestRSI)

    latestRSI = 50;

end

%% =============================================================
% INITIAL PREDICTION
% =============================================================

predictedReturn = predictInteractiveAR5( ...
    ar5Model, ...
    priceHistory, ...
    currentPrice, ...
    previousClose);

%% =============================================================
% STEP 10B THRESHOLD
% =============================================================

threshold = 0.00086042;

%% =============================================================
% INITIAL SIGNAL
% =============================================================

[signal,signalStrength] = getSignalAndStrength( ...
    predictedReturn, ...
    threshold);

fprintf('\n');
fprintf('INITIAL DASHBOARD SIGNAL\n');
fprintf('Predicted return : %+.4f%%\n', ...
    predictedReturn*100);

fprintf('Signal           : %s\n',signal);

fprintf('Signal strength  : %.2f%%\n',signalStrength);

%% =============================================================
% UI COLORS
% =============================================================

BG      = [0.025 0.035 0.055];
PANEL   = [0.055 0.070 0.105];
INPUTBG = [0.035 0.050 0.080];

TEXT  = [0.90 0.93 0.98];
MUTED = [0.48 0.56 0.68];

BLUE   = [0.08 0.55 0.95];
GREEN  = [0.20 0.85 0.45];
YELLOW = [0.95 0.75 0.18];
RED    = [0.95 0.25 0.28];

%% =============================================================
% MAIN FIGURE
% =============================================================

fig = uifigure( ...
    'Name','Crypto Sentiment Trading System', ...
    'Position',[30 30 1550 900], ...
    'Color',BG);

%% =============================================================
% HEADER
% =============================================================

header = uipanel( ...
    fig, ...
    'Position',[0 830 1550 70], ...
    'BackgroundColor',[0.035 0.050 0.080], ...
    'BorderType','none');

uilabel( ...
    header, ...
    'Position',[22 34 700 28], ...
    'Text','CRYPTO SENTIMENT TRADING', ...
    'FontSize',22, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

uilabel( ...
    header, ...
    'Position',[23 10 700 20], ...
    'Text','FINAL MACHINE LEARNING RESEARCH DASHBOARD', ...
    'FontSize',10, ...
    'FontWeight','bold', ...
    'FontColor',BLUE);

uilabel( ...
    header, ...
    'Position',[1110 30 390 22], ...
    'Text','● HISTORICAL MODEL READY', ...
    'HorizontalAlignment','right', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'FontColor',GREEN);

%% =============================================================
% MARKET INPUT PANEL
% =============================================================

inputPanel = uipanel( ...
    fig, ...
    'Position',[20 460 370 350], ...
    'Title','  MARKET INPUTS', ...
    'FontSize',13, ...
    'FontWeight','bold', ...
    'ForegroundColor',TEXT, ...
    'BackgroundColor',PANEL);

addInputLabel(inputPanel,20,290,'CURRENT PRICE');

priceEdit = uieditfield( ...
    inputPanel, ...
    'text', ...
    'Position',[20 258 330 32], ...
    'Value',sprintf('%.2f',currentPrice), ...
    'FontSize',12, ...
    'FontColor',TEXT, ...
    'BackgroundColor',INPUTBG);

addInputLabel(inputPanel,20,220,'PREVIOUS CLOSE');

previousEdit = uieditfield( ...
    inputPanel, ...
    'text', ...
    'Position',[20 188 330 32], ...
    'Value',sprintf('%.2f',previousClose), ...
    'FontSize',12, ...
    'FontColor',TEXT, ...
    'BackgroundColor',INPUTBG);

addInputLabel(inputPanel,20,150,'VOLUME');

volumeEdit = uieditfield( ...
    inputPanel, ...
    'text', ...
    'Position',[20 118 330 32], ...
    'Value',sprintf('%.2f',currentVolume), ...
    'FontSize',12, ...
    'FontColor',TEXT, ...
    'BackgroundColor',INPUTBG);

addInputLabel(inputPanel,20,80,'SENTIMENT');

sentimentEdit = uieditfield( ...
    inputPanel, ...
    'text', ...
    'Position',[20 48 160 32], ...
    'Value',sprintf('%.4f',latestSentiment), ...
    'FontSize',12, ...
    'FontColor',TEXT, ...
    'BackgroundColor',INPUTBG);

addInputLabel(inputPanel,195,80,'RSI');

rsiEdit = uieditfield( ...
    inputPanel, ...
    'text', ...
    'Position',[195 48 155 32], ...
    'Value',sprintf('%.2f',latestRSI), ...
    'FontSize',12, ...
    'FontColor',TEXT, ...
    'BackgroundColor',INPUTBG);

%% =============================================================
% BUTTON
% =============================================================

generateButton = uibutton( ...
    fig, ...
    'Position',[20 415 370 38], ...
    'Text','GENERATE TRADING SIGNAL', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'FontColor',[1 1 1], ...
    'BackgroundColor',BLUE);

%% =============================================================
% SIGNAL PANEL
% =============================================================

signalPanel = uipanel( ...
    fig, ...
    'Position',[410 460 430 350], ...
    'Title','  AI TRADING SIGNAL', ...
    'FontSize',13, ...
    'FontWeight','bold', ...
    'ForegroundColor',TEXT, ...
    'BackgroundColor',PANEL);

uilabel( ...
    signalPanel, ...
    'Position',[30 280 370 22], ...
    'Text','PREDICTED MARKET CONDITION', ...
    'HorizontalAlignment','center', ...
    'FontSize',9, ...
    'FontWeight','bold', ...
    'FontColor',MUTED);

signalLabel = uilabel( ...
    signalPanel, ...
    'Position',[30 205 370 70], ...
    'Text',signal, ...
    'HorizontalAlignment','center', ...
    'FontSize',40, ...
    'FontWeight','bold');

strengthLabel = uilabel( ...
    signalPanel, ...
    'Position',[30 165 370 28], ...
    'Text',sprintf( ...
    'Signal Strength %.1f%%', ...
    signalStrength), ...
    'HorizontalAlignment','center', ...
    'FontSize',15, ...
    'FontWeight','bold', ...
    'FontColor',TEXT);

predictionLabel = uilabel( ...
    signalPanel, ...
    'Position',[30 130 370 22], ...
    'Text',sprintf( ...
    'Predicted Return: %+.2f%%', ...
    predictedReturn*100), ...
    'HorizontalAlignment','center', ...
    'FontSize',11, ...
    'FontColor',MUTED);

uilabel( ...
    signalPanel, ...
    'Position',[30 105 370 20], ...
    'Text',sprintf( ...
    'Validation Threshold: ±%.4f%%', ...
    threshold*100), ...
    'HorizontalAlignment','center', ...
    'FontSize',10, ...
    'FontColor',MUTED);

uilabel( ...
    signalPanel, ...
    'Position',[35 45 360 45], ...
    'Text','Signal strength is NOT a probability.', ...
    'HorizontalAlignment','center', ...
    'FontSize',9, ...
    'FontColor',MUTED);

%% =============================================================
% MODEL COMPARISON PANEL
% =============================================================

modelPanel = uipanel( ...
    fig, ...
    'Position',[860 460 670 350], ...
    'Title','  FINAL MODEL COMPARISON', ...
    'FontSize',13, ...
    'FontWeight','bold', ...
    'ForegroundColor',TEXT, ...
    'BackgroundColor',PANEL);

addInfo(modelPanel,20,285, ...
    'DASHBOARD MODEL', ...
    'AR(5) RIDGE REGRESSION');

addInfo(modelPanel,20,240, ...
    'PREDICTION HORIZON', ...
    '1 DAY');

addInfo(modelPanel,20,195, ...
    'TEST PERIOD', ...
    '05-Oct-2020 to 31-Mar-2021');

addInfo(modelPanel,20,150, ...
    'AR(5) TEST RMSE', ...
    '0.042473');

addInfo(modelPanel,20,105, ...
    'AR(5) TEST RETURN', ...
    '+11.4028%');

addInfo(modelPanel,20,60, ...
    'AR(5) TEST SHARPE', ...
    '0.6646');

addInfo(modelPanel,350,285, ...
    'ORIGINAL RIDGE RETURN', ...
    '-79.8954%');

addInfo(modelPanel,350,240, ...
    'REDUCED RIDGE RETURN', ...
    '-80.1051%');

addInfo(modelPanel,350,195, ...
    'BUY & HOLD RETURN', ...
    '+443.55%');

addInfo(modelPanel,350,150, ...
    'BUY & HOLD SHARPE', ...
    '4.8243');

addInfo(modelPanel,350,105, ...
    'AR(5) MAX DRAWDOWN', ...
    '-51.69%');

addInfo(modelPanel,350,60, ...
    'TRANSACTION COST', ...
    '0.10% TURNOVER');

%% =============================================================
% PRICE HISTORY
% =============================================================

pricePanel = uipanel( ...
    fig, ...
    'Position',[20 65 600 330], ...
    'Title','  BTC PRICE HISTORY', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'ForegroundColor',TEXT, ...
    'BackgroundColor',PANEL);

axPrice = uiaxes( ...
    pricePanel, ...
    'Position',[15 15 570 275]);

axPrice.Color = BG;
axPrice.XColor = MUTED;
axPrice.YColor = MUTED;

grid(axPrice,'on');

historyStart = max(1,btcIndex-179);

historyIndices = historyStart:btcIndex;

plot( ...
    axPrice, ...
    btcDate(historyIndices), ...
    btcClose(historyIndices), ...
    'LineWidth',1.8);

title(axPrice,'Bitcoin Closing Price');

ylabel(axPrice,'USD');

%% =============================================================
% SIGNAL STRENGTH PANEL
% =============================================================

strengthPanel = uipanel( ...
    fig, ...
    'Position',[640 65 430 330], ...
    'Title','  SIGNAL STRENGTH', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'ForegroundColor',TEXT, ...
    'BackgroundColor',PANEL);

axStrength = uiaxes( ...
    strengthPanel, ...
    'Position',[35 65 360 220]);

axStrength.Color = BG;
axStrength.XColor = MUTED;
axStrength.YColor = MUTED;

grid(axStrength,'on');

updateStrengthChart( ...
    axStrength, ...
    signal, ...
    signalStrength);

uilabel( ...
    strengthPanel, ...
    'Position',[35 25 360 25], ...
    'Text','Not a statistical class-probability estimate', ...
    'HorizontalAlignment','center', ...
    'FontSize',9, ...
    'FontColor',MUTED);

%% =============================================================
% FINAL PERFORMANCE
% =============================================================

performancePanel = uipanel( ...
    fig, ...
    'Position',[1090 65 440 330], ...
    'Title','  FINAL TEST PERFORMANCE', ...
    'FontSize',12, ...
    'FontWeight','bold', ...
    'ForegroundColor',TEXT, ...
    'BackgroundColor',PANEL);

createMetric(performancePanel,20,250, ...
    'AR(5) RMSE','0.042473');

createMetric(performancePanel,225,250, ...
    'AR(5) MAE','0.030789');

createMetric(performancePanel,20,185, ...
    'AR(5) R²','-0.073428');

createMetric(performancePanel,225,185, ...
    'AR(5) RETURN','+11.4028%');

createMetric(performancePanel,20,120, ...
    'AR(5) SHARPE','0.6646');

createMetric(performancePanel,225,120, ...
    'AR(5) MAX DD','-51.69%');

createMetric(performancePanel,20,55, ...
    'B&H RETURN','+443.55%');

createMetric(performancePanel,225,55, ...
    'B&H SHARPE','4.8243');

%% =============================================================
% COMPARISON TABLE
% =============================================================

comparisonPanel = uipanel( ...
    fig, ...
    'Position',[20 0 1510 58], ...
    'Title','  FINAL STEP 10B MODEL COMPARISON', ...
    'FontSize',10, ...
    'FontWeight','bold', ...
    'ForegroundColor',TEXT, ...
    'BackgroundColor',PANEL);

comparisonData = {
    'AR(5)'           '0.04247' '+11.40%'  '0.665'  '-51.69%'
    'Original Ridge'  '0.07264' '-79.90%'  '-3.705' '-82.87%'
    'Reduced Ridge'   '0.08050' '-80.11%'  '-3.781' '-82.09%'
    'Buy & Hold'      '—'       '+443.55%' '4.824'  '-25.17%'
    };

uitable( ...
    comparisonPanel, ...
    'Data',comparisonData, ...
    'ColumnName',{ ...
    'Model', ...
    'RMSE', ...
    'Return', ...
    'Sharpe', ...
    'Max Drawdown'}, ...
    'Position',[10 2 1490 35], ...
    'FontSize',9);

%% =============================================================
% FOOTER
% =============================================================

footer = uipanel( ...
    fig, ...
    'Position',[0 0 1550 35], ...
    'BackgroundColor',[0.018 0.025 0.040], ...
    'BorderType','none');

uilabel( ...
    footer, ...
    'Position',[20 5 1510 20], ...
    'Text','ACADEMIC RESEARCH PROTOTYPE  |  HISTORICAL DATA  |  NOT A LIVE TRADING SYSTEM', ...
    'HorizontalAlignment','center', ...
    'FontSize',9, ...
    'FontWeight','bold', ...
    'FontColor',MUTED);

%% =============================================================
% INITIAL COLOR
% =============================================================

updateSignalAppearance( ...
    signalLabel, ...
    signal, ...
    GREEN, ...
    YELLOW, ...
    RED);

%% =============================================================
% BUTTON CALLBACK
% =============================================================

generateButton.ButtonPushedFcn = @generateSignal;

%% =============================================================
% CONSOLE SUMMARY
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 DASHBOARD READY\n');
fprintf('============================================================\n');

fprintf('Dashboard Date   : %s\n', ...
    datestr(commonDate,'dd-mmm-yyyy'));

fprintf('Current Price    : $%.2f\n',currentPrice);

fprintf('Previous Close   : $%.2f\n',previousClose);

fprintf('Volume           : %.2f\n',currentVolume);

fprintf('Sentiment        : %.4f\n',latestSentiment);

fprintf('RSI              : %.2f\n',latestRSI);

fprintf('Predicted Return : %+.4f%%\n', ...
    predictedReturn*100);

fprintf('Signal           : %s\n',signal);

fprintf('Signal Strength  : %.2f%%\n',signalStrength);

fprintf('Threshold        : ±%.4f%%\n', ...
    threshold*100);

fprintf('============================================================\n');

%% =============================================================
% CALLBACK
% =============================================================

    function generateSignal(~,~)

        %% Read inputs

        p = str2double(priceEdit.Value);

        prev = str2double(previousEdit.Value);

        vol = str2double(volumeEdit.Value);

        sent = str2double(sentimentEdit.Value);

        rsiInput = str2double(rsiEdit.Value);

        %% Validate

        if ~isfinite(p) || p <= 0

            uialert( ...
                fig, ...
                'Enter a valid positive Current Price.', ...
                'Invalid Input');

            return;

        end

        if ~isfinite(prev) || prev <= 0

            uialert( ...
                fig, ...
                'Enter a valid positive Previous Close.', ...
                'Invalid Input');

            return;

        end

        if ~isfinite(vol)
            vol = currentVolume;
        end

        if ~isfinite(sent)
            sent = latestSentiment;
        end

        if ~isfinite(rsiInput)
            rsiInput = latestRSI;
        end

        %% =====================================================
        % INTERACTIVE AR(5) PREDICTION
        % =====================================================

        pred = predictInteractiveAR5( ...
            ar5Model, ...
            priceHistory, ...
            p, ...
            prev);

        if ~isfinite(pred)

            uialert( ...
                fig, ...
                'AR(5) prediction could not be generated.', ...
                'Prediction Error');

            return;

        end

        %% =====================================================
        % SIGNAL
        % =====================================================

        [newSignal,newStrength] = ...
            getSignalAndStrength( ...
            pred, ...
            threshold);

        %% Update UI

        signalLabel.Text = newSignal;

        strengthLabel.Text = ...
            sprintf( ...
            'Signal Strength %.1f%%', ...
            newStrength);

        predictionLabel.Text = ...
            sprintf( ...
            'Predicted Return: %+.2f%%', ...
            pred*100);

        updateStrengthChart( ...
            axStrength, ...
            newSignal, ...
            newStrength);

        updateSignalAppearance( ...
            signalLabel, ...
            newSignal, ...
            GREEN, ...
            YELLOW, ...
            RED);

        %% =====================================================
        % CONSOLE
        % =====================================================

        interactiveReturn = ...
            (p-prev)/prev;

        fprintf('\n');
        fprintf('----- NEW INTERACTIVE SIGNAL -----\n');

        fprintf( ...
            'Dashboard Date: %s\n', ...
            datestr(commonDate,'dd-mmm-yyyy'));

        fprintf( ...
            'Price       : %.2f\n',p);

        fprintf( ...
            'Previous    : %.2f\n',prev);

        fprintf( ...
            'Input Return : %+.4f%%\n', ...
            interactiveReturn*100);

        fprintf( ...
            'Volume      : %.2f\n',vol);

        fprintf( ...
            'Sentiment   : %.4f\n',sent);

        fprintf( ...
            'RSI         : %.2f\n',rsiInput);

        fprintf( ...
            'AR(5)       : %+.4f%%\n', ...
            pred*100);

        fprintf( ...
            'Signal      : %s\n', ...
            newSignal);

        fprintf( ...
            'Strength    : %.2f%%\n', ...
            newStrength);

        fprintf( ...
            'NOTE        : Strength is not probability.\n');

        fprintf( ...
            '---------------------------------\n');

    end

end

%% =============================================================
% BUILD AR(5) MODEL
% =============================================================

function model = buildAR5Model( ...
    dates, ...
    closePrice)

%% -------------------------------------------------------------
% Returns
% -------------------------------------------------------------

returns = ...
    [NaN; ...
    diff(closePrice)./closePrice(1:end-1)];

%% -------------------------------------------------------------
% Construct AR(5)
%
% X(t) = [r(t-1), r(t-2), ..., r(t-5)]
% y(t) = r(t)
% -------------------------------------------------------------

n = numel(returns);

X = NaN(n,5);

Y = NaN(n,1);

rowDates = NaT(n,1);

for t = 6:n

    X(t,:) = ...
        [returns(t-1), ...
         returns(t-2), ...
         returns(t-3), ...
         returns(t-4), ...
         returns(t-5)];

    Y(t) = returns(t);

    rowDates(t) = dates(t);

end

valid = ...
    all(isfinite(X),2) & ...
    isfinite(Y) & ...
    ~isnat(rowDates);

X = X(valid,:);
Y = Y(valid);
rowDates = rowDates(valid);

%% -------------------------------------------------------------
% Fixed chronological periods
% -------------------------------------------------------------

trainStart = datetime(2018,1,6);

trainEnd = datetime(2020,4,10);

valStart = datetime(2020,4,11);

valEnd = datetime(2020,10,4);

testStart = datetime(2020,10,5);

testEnd = datetime(2021,3,31);

trainMask = ...
    rowDates >= trainStart & ...
    rowDates <= trainEnd;

valMask = ...
    rowDates >= valStart & ...
    rowDates <= valEnd;

testMask = ...
    rowDates >= testStart & ...
    rowDates <= testEnd;

Xtrain = X(trainMask,:);
Ytrain = Y(trainMask);

Xval = X(valMask,:);
Yval = Y(valMask);

Xtest = X(testMask,:);
Ytest = Y(testMask);

%% -------------------------------------------------------------
% Train-only standardization
% -------------------------------------------------------------

mu = mean(Xtrain,1);

sigma = std(Xtrain,0,1);

sigma(sigma == 0 | ~isfinite(sigma)) = 1;

XtrainZ = ...
    (Xtrain-mu)./sigma;

XvalZ = ...
    (Xval-mu)./sigma;

XtestZ = ...
    (Xtest-mu)./sigma;

%% -------------------------------------------------------------
% Ridge
% -------------------------------------------------------------

lambda = 1e-4;

try

    mdl = fitrlinear( ...
        XtrainZ, ...
        Ytrain, ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda',lambda, ...
        'FitBias',true);

    predVal = predict(mdl,XvalZ);

catch

    % Manual ridge fallback

    X1 = [ones(size(XtrainZ,1),1),XtrainZ];

    I = eye(size(X1,2));

    I(1,1) = 0;

    beta = ...
        (X1'*X1 + lambda*I) \ ...
        (X1'*Ytrain);

    mdl = beta;

    predVal = ...
        [ones(size(XvalZ,1),1),XvalZ] * beta;

end

%% -------------------------------------------------------------
% Validation metrics
% -------------------------------------------------------------

validationRMSE = sqrt( ...
    mean((predVal-Yval).^2));

validationMAE = mean( ...
    abs(predVal-Yval));

%% -------------------------------------------------------------
% Final refit on train + validation
% -------------------------------------------------------------

Xrefit = [Xtrain;Xval];

Yrefit = [Ytrain;Yval];

%% Important:
% Recalculate standardization using Train + Validation for final
% model refit, because the final model is allowed to use
% development data before the untouched test.

muFinal = mean(Xrefit,1);

sigmaFinal = std(Xrefit,0,1);

sigmaFinal( ...
    sigmaFinal == 0 | ...
    ~isfinite(sigmaFinal)) = 1;

XrefitZ = ...
    (Xrefit-muFinal)./sigmaFinal;

XtestFinalZ = ...
    (Xtest-muFinal)./sigmaFinal;

try

    finalModel = fitrlinear( ...
        XrefitZ, ...
        Yrefit, ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda',lambda, ...
        'FitBias',true);

catch

    X1 = [ones(size(XrefitZ,1),1),XrefitZ];

    I = eye(size(X1,2));

    I(1,1) = 0;

    beta = ...
        (X1'*X1 + lambda*I) \ ...
        (X1'*Yrefit);

    finalModel = beta;

end

%% -------------------------------------------------------------
% Store everything required by dashboard
% -------------------------------------------------------------

model.lambda = lambda;

model.model = finalModel;

model.mu = muFinal;

model.sigma = sigmaFinal;

model.trainMu = mu;

model.trainSigma = sigma;

model.validationRMSE = validationRMSE;

model.validationMAE = validationMAE;

model.trainX = Xtrain;

model.trainY = Ytrain;

model.valX = Xval;

model.valY = Yval;

model.testX = Xtest;

model.testY = Ytest;

model.refitX = Xrefit;

model.refitY = Yrefit;

model.testDates = rowDates(testMask);

%% -------------------------------------------------------------
% Console information
% -------------------------------------------------------------

fprintf('\n');
fprintf('AR(5) model built successfully.\n');

fprintf( ...
    'Train observations       : %d\n', ...
    numel(Ytrain));

fprintf( ...
    'Validation observations  : %d\n', ...
    numel(Yval));

fprintf( ...
    'Test observations        : %d\n', ...
    numel(Ytest));

fprintf( ...
    'Lambda                   : %.6f\n', ...
    lambda);

end

%% =============================================================
% INTERACTIVE AR(5) PREDICTION
% =============================================================

function prediction = predictInteractiveAR5( ...
    model, ...
    historicalPrices, ...
    currentPrice, ...
    previousClose)

prediction = NaN;

%% -------------------------------------------------------------
% Build historical returns
% -------------------------------------------------------------

returns = ...
    [NaN; ...
    diff(historicalPrices)./ ...
    historicalPrices(1:end-1)];

%% -------------------------------------------------------------
% Replace latest return with GUI input
% -------------------------------------------------------------

latestReturn = ...
    (currentPrice-previousClose)/previousClose;

returns(end) = latestReturn;

%% -------------------------------------------------------------
% Need five lagged returns for AR(5)
%
% Model was trained as:
%
% [r(t-1) r(t-2) r(t-3) r(t-4) r(t-5)]
%
% -------------------------------------------------------------

if numel(returns) < 6

    return;

end

x = [ ...
    returns(end), ...
    returns(end-1), ...
    returns(end-2), ...
    returns(end-3), ...
    returns(end-4)];

%% -------------------------------------------------------------
% Standardize using final development-data statistics
% -------------------------------------------------------------

xZ = ...
    (x-model.mu)./model.sigma;

%% -------------------------------------------------------------
% Predict
% -------------------------------------------------------------

try

    if isobject(model.model)

        prediction = predict( ...
            model.model, ...
            xZ);

    else

        beta = model.model(:);

        prediction = ...
            [1,xZ] * beta;

    end

catch

    %% Manual fallback for RegressionLinear object

    try

        if isprop(model.model,'Beta') && ...
                isprop(model.model,'Bias')

            beta = model.model.Beta(:);

            bias = model.model.Bias;

            prediction = ...
                xZ*beta + bias;

        end

    catch

        prediction = NaN;

    end

end

prediction = double(prediction);

end

%% =============================================================
% SIGNAL + STRENGTH
% =============================================================

function [signal,strength] = ...
    getSignalAndStrength( ...
    predictedReturn, ...
    threshold)

if predictedReturn >= threshold

    signal = 'BUY';

elseif predictedReturn <= -threshold

    signal = 'SELL';

else

    signal = 'HOLD';

end

strength = ...
    100 * ...
    abs(predictedReturn) / ...
    max(threshold,eps);

strength = min(strength,100);

end

%% =============================================================
% SIGNAL COLOR
% =============================================================

function updateSignalAppearance( ...
    label, ...
    signal, ...
    GREEN, ...
    YELLOW, ...
    RED)

switch upper(signal)

    case 'BUY'

        label.FontColor = GREEN;

    case 'SELL'

        label.FontColor = RED;

    otherwise

        label.FontColor = YELLOW;

end

end

%% =============================================================
% STRENGTH CHART
% =============================================================

function updateStrengthChart( ...
    ax, ...
    signal, ...
    strength)

values = [0 0 0];

switch upper(signal)

    case 'BUY'
        values(1) = strength;

    case 'HOLD'
        values(2) = strength;

    case 'SELL'
        values(3) = strength;

end

cla(ax);

bar( ...
    ax, ...
    categorical({'BUY','HOLD','SELL'}), ...
    values);

ylim(ax,[0 100]);

ylabel(ax,'Strength (%)');

title(ax,'AR(5) Prediction Strength');

grid(ax,'on');

end

%% =============================================================
% INPUT LABEL
% =============================================================

function addInputLabel( ...
    parent, ...
    x, ...
    y, ...
    textValue)

uilabel( ...
    parent, ...
    'Position',[x y 150 20], ...
    'Text',textValue, ...
    'FontSize',9, ...
    'FontWeight','bold', ...
    'FontColor',[0.48 0.56 0.68]);

end

%% =============================================================
% INFORMATION LABEL
% =============================================================

function addInfo( ...
    parent, ...
    x, ...
    y, ...
    name, ...
    value)

uilabel( ...
    parent, ...
    'Position',[x y 250 17], ...
    'Text',name, ...
    'FontSize',8, ...
    'FontWeight','bold', ...
    'FontColor',[0.45 0.53 0.65]);

uilabel( ...
    parent, ...
    'Position',[x y-25 300 24], ...
    'Text',value, ...
    'FontSize',11, ...
    'FontWeight','bold', ...
    'FontColor',[0.90 0.93 0.98]);

end

%% =============================================================
% PERFORMANCE METRIC
% =============================================================

function createMetric( ...
    parent, ...
    x, ...
    y, ...
    name, ...
    value)

uilabel( ...
    parent, ...
    'Position',[x y 180 18], ...
    'Text',name, ...
    'FontSize',8, ...
    'FontWeight','bold', ...
    'FontColor',[0.45 0.53 0.65]);

uilabel( ...
    parent, ...
    'Position',[x y-28 190 28], ...
    'Text',value, ...
    'FontSize',15, ...
    'FontWeight','bold', ...
    'FontColor',[0.90 0.93 0.98]);

end

%% =============================================================
% BTC CSV READER
% =============================================================

function [d,o,h,l,c,v] = readBTCraw(fileName)

fprintf('\nReading raw BTC CSV...\n');

raw = readcell( ...
    fileName, ...
    'Delimiter',',');

[nRows,nCols] = size(raw);

fprintf( ...
    'Raw BTC CSV dimensions: %d rows x %d columns\n', ...
    nRows,nCols);

header = raw(1,:);

fprintf('\nBTC CSV HEADER:\n');

for k = 1:nCols

    fprintf( ...
        '  Column %d : %s\n', ...
        k, ...
        char(string(header{k})));

end

data = raw(2:end,:);

n = size(data,1);

d = NaT(n,1);

o = NaN(n,1);

h = NaN(n,1);

l = NaN(n,1);

c = NaN(n,1);

v = NaN(n,1);

for i = 1:n

    d(i) = parseDateRobust(data{i,1});

    o(i) = parseNumber(data{i,2});

    h(i) = parseNumber(data{i,3});

    l(i) = parseNumber(data{i,4});

    c(i) = parseNumber(data{i,5});

    v(i) = parseNumber(data{i,6});

end

fprintf('\nBTC RAW COLUMN DIAGNOSTICS:\n');

fprintf( ...
    '  Date valid   : %d / %d\n', ...
    sum(~isnat(d)),n);

fprintf( ...
    '  Open valid   : %d / %d\n', ...
    sum(isfinite(o)),n);

fprintf( ...
    '  High valid   : %d / %d\n', ...
    sum(isfinite(h)),n);

fprintf( ...
    '  Low valid    : %d / %d\n', ...
    sum(isfinite(l)),n);

fprintf( ...
    '  Close valid  : %d / %d\n', ...
    sum(isfinite(c)),n);

fprintf( ...
    '  Volume valid : %d / %d\n', ...
    sum(isfinite(v)),n);

fprintf('\nBTC DATE EXAMPLES:\n');

for i = 1:min(3,n)

    fprintf( ...
        '  Row %d raw value: %s\n', ...
        i, ...
        char(string(data{i,1})));

    if ~isnat(d(i))

        fprintf( ...
            '       parsed    : %s\n', ...
            datestr(d(i),'yyyy-mm-dd HH:MM:SS'));

    else

        fprintf( ...
            '       parsed    : INVALID\n');

    end

end

if sum(~isnat(d)) == 0

    error('No valid BTC dates found.');

end

fprintf('\nBTC raw CSV extraction successful.\n');

end

%% =============================================================
% DATE PARSER
% =============================================================

function d = parseDateRobust(x)

d = NaT;

if isempty(x)
    return;
end

if isdatetime(x)

    d = x(1);

    return;

end

if isnumeric(x)

    if isscalar(x) && isfinite(x)

        d = convertTimestamp(double(x));

    end

    return;

end

try

    s = strtrim(string(x));

catch

    return;

end

if isempty(s) || ismissing(s) || s == ""

    return;

end

%% Remove UTC/GMT

s = regexprep( ...
    s, ...
    '\s+(UTC|GMT)$', ...
    '', ...
    'ignorecase');

s = strtrim(s);

%% Exact formats

formats = { ...
    'yyyy-MM-dd HH:mm:ss.SSSSSS'
    'yyyy-MM-dd HH:mm:ss.SSS'
    'yyyy-MM-dd HH:mm:ss'
    'yyyy-MM-dd HH:mm'
    'yyyy-MM-dd'
    'yyyy/MM/dd HH:mm:ss.SSSSSS'
    'yyyy/MM/dd HH:mm:ss.SSS'
    'yyyy/MM/dd HH:mm:ss'
    'yyyy/MM/dd HH:mm'
    'yyyy/MM/dd'
    'dd-MM-yyyy HH:mm:ss.SSSSSS'
    'dd-MM-yyyy HH:mm:ss.SSS'
    'dd-MM-yyyy HH:mm:ss'
    'dd-MM-yyyy HH:mm'
    'dd-MM-yyyy'
    'dd/MM/yyyy HH:mm:ss.SSSSSS'
    'dd/MM/yyyy HH:mm:ss.SSS'
    'dd/MM/yyyy HH:mm:ss'
    'dd/MM/yyyy HH:mm'
    'dd/MM/yyyy'
    };

for k = 1:numel(formats)

    try

        temp = datetime( ...
            s, ...
            'InputFormat',formats{k});

        if ~isnat(temp)

            d = temp;

            return;

        end

    catch

    end

end

%% Automatic

try

    temp = datetime(s);

    if ~isnat(temp)

        d = temp;

        return;

    end

catch

end

%% Numeric text

value = str2double(s);

if isfinite(value)

    d = convertTimestamp(value);

    if ~isnat(d)

        return;

    end

end

end

%% =============================================================
% TIMESTAMP CONVERSION
% =============================================================

function d = convertTimestamp(value)

d = NaT;

if ~isfinite(value)

    return;

end

try

    % milliseconds

    if value >= 1e11 && value < 1e14

        d = datetime( ...
            value/1000, ...
            'ConvertFrom','posixtime');

        return;

    end

    % microseconds

    if value >= 1e14 && value < 1e17

        d = datetime( ...
            value/1e6, ...
            'ConvertFrom','posixtime');

        return;

    end

    % seconds

    if value >= 1e9 && value < 1e11

        d = datetime( ...
            value, ...
            'ConvertFrom','posixtime');

        return;

    end

    % MATLAB datenum

    if value >= 700000 && value < 900000

        d = datetime( ...
            value, ...
            'ConvertFrom','datenum');

        return;

    end

    % Excel

    if value >= 20000 && value < 100000

        d = datetime( ...
            value, ...
            'ConvertFrom','excel');

    end

catch

    d = NaT;

end

end

%% =============================================================
% NUMBER PARSER
% =============================================================

function n = parseNumber(x)

n = NaN;

if isempty(x)

    return;

end

if isnumeric(x)

    if isscalar(x) && isfinite(x)

        n = double(x);

    end

    return;

end

try

    s = strtrim(string(x));

catch

    return;

end

if isempty(s) || ismissing(s)

    return;

end

s = erase(s,',');
s = erase(s,'$');
s = erase(s,'%');

n = str2double(s);

if isfinite(n)

    return;

end

token = regexp( ...
    char(s), ...
    '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?', ...
    'match');

if ~isempty(token)

    n = str2double(token{1});

end

end

%% =============================================================
% SENTIMENT LOOKUP
% =============================================================

function value = getSentimentAtDate( ...
    dates, ...
    values, ...
    targetDate)

value = 0;

targetDay = dateshift( ...
    targetDate, ...
    'start','day');

dataDays = dateshift( ...
    dates, ...
    'start','day');

idx = find( ...
    dataDays == targetDay, ...
    1, ...
    'last');

if ~isempty(idx)

    value = values(idx);

    return;

end

idx = find( ...
    dataDays <= targetDay, ...
    1, ...
    'last');

if ~isempty(idx)

    value = values(idx);

end

end

%% =============================================================
% CONVERT DATES
% =============================================================

function d = convertDates(x)

d = NaT(numel(x),1);

for i = 1:numel(x)

    d(i) = parseDateRobust(x(i));

end

end

%% =============================================================
% CONVERT NUMBERS
% =============================================================

function n = convertNumbers(x)

n = NaN(numel(x),1);

for i = 1:numel(x)

    n(i) = parseNumber(x(i));

end

end

%% =============================================================
% FIND COLUMN
% =============================================================

function idx = findColumn(T,candidates)

idx = [];

vars = T.Properties.VariableNames;

for i = 1:numel(candidates)

    hit = strcmpi(vars,candidates{i});

    if any(hit)

        idx = find(hit,1);

        return;

    end

end

for i = 1:numel(vars)

    a = lower(regexprep( ...
        vars{i}, ...
        '[^a-zA-Z0-9]',''));

    for j = 1:numel(candidates)

        b = lower(regexprep( ...
            candidates{j}, ...
            '[^a-zA-Z0-9]',''));

        if strcmp(a,b)

            idx = i;

            return;

        end

    end

end

end

%% =============================================================
% RSI
% =============================================================

function rsi = calculateRSI(price,period)

price = price(:);

rsi = NaN(size(price));

if numel(price) < period+1

    return;

end

delta = [NaN;diff(price)];

gain = max(delta,0);

loss = max(-delta,0);

gain(1) = 0;

loss(1) = 0;

avgGain = movmean( ...
    gain, ...
    [period-1 0], ...
    'omitnan');

avgLoss = movmean( ...
    loss, ...
    [period-1 0], ...
    'omitnan');

valid = avgLoss > 0;

rs = avgGain(valid)./avgLoss(valid);

rsi(valid) = ...
    100 - 100./(1+rs);

rsi( ...
    avgLoss == 0 & ...
    avgGain > 0) = 100;

rsi( ...
    avgGain == 0 & ...
    avgLoss > 0) = 0;

end