function step1_data_loading_and_eda()
%STEP1_DATA_LOADING_AND_EDA
% Load, clean, align BTC price and sentiment data and perform EDA.
%
% Output:
%   data/processed/merged_dataset.csv
%   results/figures/eda_price_and_sentiment.png

clc;
close all;

fprintf('\n============================================================\n');
fprintf(' [STEP 1] DATA LOADING, MERGING & EXPLORATORY DATA ANALYSIS\n');
fprintf('============================================================\n\n');

%% -----------------------------------------------------------
% 1. SET UP PROJECT PATHS
% ------------------------------------------------------------

thisFile = mfilename('fullpath');

if isempty(thisFile)
    srcDir = pwd;
else
    srcDir = fileparts(thisFile);
end

projectDir = fileparts(srcDir);
utilsDir   = fullfile(srcDir, 'utils');

% Add project folders to MATLAB path
addpath(srcDir);
addpath(utilsDir);

fprintf('Project directory:\n%s\n\n', projectDir);

%% -----------------------------------------------------------
% 2. INITIALIZE UTILITY LIBRARIES
% ------------------------------------------------------------

try
    dUtils = dataUtils();
    pUtils = plotUtils();
catch ME
    error('step1:utilities', ...
        ['Unable to initialize utility libraries.\n' ...
         'Make sure these files exist:\n' ...
         '  src/utils/dataUtils.m\n' ...
         '  src/utils/plotUtils.m\n\n' ...
         'Original error: %s'], ME.message);
end

%% -----------------------------------------------------------
% 3. DEFINE INPUT AND OUTPUT FILE PATHS
% ------------------------------------------------------------

rawBtcFile = fullfile( ...
    projectDir, ...
    'data', ...
    'raw', ...
    'btc_1d_data_2018_to_2025.csv');

rawSentFile = fullfile( ...
    projectDir, ...
    'data', ...
    'raw', ...
    'sentiment_data_consolidated.csv');

procDir = fullfile( ...
    projectDir, ...
    'data', ...
    'processed');

figDir = fullfile( ...
    projectDir, ...
    'results', ...
    'figures');

% Create output directories if they do not exist
if ~exist(procDir, 'dir')
    mkdir(procDir);
end

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

%% -----------------------------------------------------------
% 4. CHECK INPUT FILES
% ------------------------------------------------------------

if ~isfile(rawBtcFile)
    error('step1:missingBTC', ...
        'BTC dataset was not found:\n%s', rawBtcFile);
end

if ~isfile(rawSentFile)
    error('step1:missingSentiment', ...
        'Sentiment dataset was not found:\n%s', rawSentFile);
end

%% -----------------------------------------------------------
% 5. LOAD BTC DATA
% ------------------------------------------------------------

fprintf('-> Loading BTC OHLCV data...\n');
fprintf('   File: %s\n\n', rawBtcFile);

try
    btcTable = dUtils.loadBTCData(rawBtcFile);
catch ME
    error('step1:btcLoad', ...
        ['Failed to load BTC data.\n' ...
         'Check the CSV format and column names.\n\n' ...
         'Original error: %s'], ME.message);
end

if isempty(btcTable) || height(btcTable) == 0
    error('step1:emptyBTC', ...
        'BTC dataset is empty after loading.');
end

dUtils.printDataSummary(btcTable, 'BTC Daily OHLCV');

%% -----------------------------------------------------------
% 6. LOAD SENTIMENT DATA
% ------------------------------------------------------------

fprintf('-> Loading daily sentiment data...\n');
fprintf('   File: %s\n\n', rawSentFile);

try
    sentTable = dUtils.loadSentimentData(rawSentFile);
catch ME
    error('step1:sentimentLoad', ...
        ['Failed to load sentiment data.\n' ...
         'Check the CSV format and column names.\n\n' ...
         'Original error: %s'], ME.message);
end

if isempty(sentTable) || height(sentTable) == 0
    error('step1:emptySentiment', ...
        'Sentiment dataset is empty after loading.');
end

dUtils.printDataSummary(sentTable, 'Daily Sentiment Feed');

%% -----------------------------------------------------------
% 7. MERGE BTC AND SENTIMENT DATA
% ------------------------------------------------------------

fprintf('-> Inner-joining BTC and sentiment datasets on Date...\n');

try
    merged = dUtils.mergeDatasets(btcTable, sentTable);
catch ME
    error('step1:merge', ...
        ['Failed to merge BTC and sentiment datasets.\n' ...
         'Both datasets must contain a valid Date field.\n\n' ...
         'Original error: %s'], ME.message);
end

if isempty(merged) || height(merged) == 0
    error('step1:noOverlap', ...
        ['No overlapping dates were found between the BTC and ' ...
         'sentiment datasets.']);
end

dUtils.printDataSummary(merged, ...
    'Merged BTC Price & Sentiment Dataset');

if height(merged) < 250
    warning('step1:smallOverlap', ...
        ['Fewer than 250 overlapping days were found. ' ...
         'The model may not have enough data.']);
end

%% -----------------------------------------------------------
% 8. BASIC DATA VALIDATION
% ------------------------------------------------------------

requiredVariables = {'Date', 'Close', 'Sentiment'};

for i = 1:numel(requiredVariables)

    if ~ismember(requiredVariables{i}, merged.Properties.VariableNames)

        error('step1:missingVariable', ...
            'Required variable "%s" is missing from merged dataset.', ...
            requiredVariables{i});

    end
end

% Sort chronologically
merged = sortrows(merged, 'Date');

% Remove duplicate dates if any
[~, uniqueIdx] = unique(merged.Date, 'stable');

if numel(uniqueIdx) < height(merged)

    warning('step1:duplicateDates', ...
        'Duplicate dates detected. Keeping the first occurrence.');

    merged = merged(uniqueIdx, :);

end

% Check for invalid close prices
invalidClose = ~isfinite(merged.Close) | merged.Close <= 0;

if any(invalidClose)

    warning('step1:invalidClose', ...
        '%d rows contain invalid BTC Close prices and will be removed.', ...
        sum(invalidClose));

    merged(invalidClose, :) = [];

end

% Check again after cleaning
if isempty(merged) || height(merged) < 2

    error('step1:insufficientData', ...
        'Insufficient valid observations remain after data cleaning.');

end

%% -----------------------------------------------------------
% 9. CALCULATE DAILY RETURNS
% ------------------------------------------------------------

closePrices = merged.Close;

returns = nan(height(merged), 1);

returns(2:end) = ...
    (closePrices(2:end) - closePrices(1:end-1)) ...
    ./ closePrices(1:end-1);

%% -----------------------------------------------------------
% 10. CORRELATION BETWEEN SENTIMENT AND RETURNS
% ------------------------------------------------------------

validCorrelationRows = ...
    isfinite(returns) & ...
    isfinite(merged.Sentiment);

if sum(validCorrelationRows) >= 3

    corrVal = corr( ...
        returns(validCorrelationRows), ...
        merged.Sentiment(validCorrelationRows), ...
        'Rows', 'complete');

else

    corrVal = NaN;

end

%% -----------------------------------------------------------
% 11. EDA STATISTICS
% ------------------------------------------------------------

meanClose = mean(merged.Close, 'omitnan');
meanSentiment = mean(merged.Sentiment, 'omitnan');
stdSentiment = std(merged.Sentiment, 0, 'omitnan');

validReturns = returns(isfinite(returns));

if isempty(validReturns)

    meanReturn = NaN;
    stdReturn = NaN;

else

    meanReturn = mean(validReturns, 'omitnan');
    stdReturn = std(validReturns, 0, 'omitnan');

end

fprintf('------------------------------------------------------------\n');
fprintf(' [EDA KEY FINDINGS]\n');
fprintf('------------------------------------------------------------\n');

fprintf(' Overlapping Days       : %d\n', height(merged));

fprintf(' Start Date             : %s\n', ...
    datestr(merged.Date(1), 'yyyy-mm-dd'));

fprintf(' End Date               : %s\n', ...
    datestr(merged.Date(end), 'yyyy-mm-dd'));

fprintf(' Mean BTC Close Price   : $%.2f\n', ...
    meanClose);

fprintf(' Mean Sentiment Score   : %.4f\n', ...
    meanSentiment);

fprintf(' Sentiment Std. Dev.    : %.4f\n', ...
    stdSentiment);

fprintf(' Mean 1-Day Return      : %.6f\n', ...
    meanReturn);

fprintf(' Return Std. Dev.       : %.6f\n', ...
    stdReturn);

if isfinite(corrVal)

    fprintf(' Return/Sentiment Corr. : %.4f\n', ...
        corrVal);

else

    fprintf(' Return/Sentiment Corr. : NaN\n');

end

fprintf('------------------------------------------------------------\n\n');

%% -----------------------------------------------------------
% 12. GENERATE EDA VISUALIZATION
% ------------------------------------------------------------

edaFigPath = fullfile( ...
    figDir, ...
    'eda_price_and_sentiment.png');

fprintf('-> Generating EDA visualization...\n');
fprintf('   Output: %s\n', edaFigPath);

try

    pUtils.plotEDA( ...
        merged.Date, ...
        merged.Close, ...
        merged.Sentiment, ...
        edaFigPath);

catch ME

    warning('step1:plotFailed', ...
        ['EDA plot could not be generated.\n' ...
         'The processed dataset will still be saved.\n\n' ...
         'Original error: %s'], ME.message);

end

%% -----------------------------------------------------------
% 13. SAVE PROCESSED DATASET
% ------------------------------------------------------------

outFile = fullfile( ...
    procDir, ...
    'merged_dataset.csv');

fprintf('-> Saving processed merged dataset...\n');
fprintf('   Output: %s\n', outFile);

try

    writetable(merged, outFile);

catch ME

    error('step1:saveFailed', ...
        ['Unable to save merged dataset.\n' ...
         'Check that the output directory is writable.\n\n' ...
         'Original error: %s'], ME.message);

end

%% -----------------------------------------------------------
% 14. FINAL VALIDATION
% ------------------------------------------------------------

if ~isfile(outFile)

    error('step1:outputMissing', ...
        'Processed dataset was not created successfully.');

end

fprintf('\n============================================================\n');
fprintf(' [STEP 1 COMPLETE]\n');
fprintf('============================================================\n');

fprintf('Processed observations : %d\n', height(merged));
fprintf('Processed dataset      : %s\n', outFile);
fprintf('EDA figure             : %s\n', edaFigPath);

fprintf('\nProceeding to Step 2...\n\n');

end