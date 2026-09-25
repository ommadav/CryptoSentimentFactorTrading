function step2_sentiment_classification()
%STEP2_SENTIMENT_CLASSIFICATION
% Classify continuous sentiment into interpretable sentiment regimes.
%
% Input:
%   data/processed/merged_dataset.csv
%
% Output:
%   data/processed/sentiment_classified_dataset.csv
%   results/figures/sentiment_classification.png
%
% Sentiment regimes:
%   Negative
%   Neutral
%   Positive
%
% The thresholds are estimated using the training portion only to
% avoid future-information leakage.

clc;
close all;

fprintf('\n============================================================\n');
fprintf(' [STEP 2] SENTIMENT CLASSIFICATION\n');
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
    'merged_dataset.csv');

outputFile = fullfile( ...
    projectDir, ...
    'data', ...
    'processed', ...
    'sentiment_classified_dataset.csv');

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

    error('step2:missingInput', ...
        ['Merged dataset was not found:\n%s\n\n' ...
         'Run Step 1 before Step 2.'], ...
        inputFile);

end

%% -----------------------------------------------------------
% 4. LOAD MERGED DATASET
% ------------------------------------------------------------

fprintf('-> Loading merged dataset...\n');
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

        error('step2:readError', ...
            'Unable to read merged dataset.\n%s', ...
            ME.message);

    end

end

if isempty(data) || height(data) == 0

    error('step2:emptyData', ...
        'Merged dataset contains no observations.');

end

%% -----------------------------------------------------------
% 5. VALIDATE REQUIRED COLUMNS
% ------------------------------------------------------------

requiredColumns = ...
    {'Date', 'Sentiment'};

for k = 1:numel(requiredColumns)

    if ~ismember( ...
            requiredColumns{k}, ...
            data.Properties.VariableNames)

        error('step2:missingColumn', ...
            'Required column "%s" is missing.', ...
            requiredColumns{k});

    end

end

%% -----------------------------------------------------------
% 6. SORT DATA CHRONOLOGICALLY
% ------------------------------------------------------------

if ~isdatetime(data.Date)

    try

        data.Date = datetime(data.Date);

    catch ME

        error('step2:dateError', ...
            'Unable to convert Date column to datetime.\n%s', ...
            ME.message);

    end

end

data = sortrows(data, 'Date');

%% -----------------------------------------------------------
% 7. REMOVE INVALID SENTIMENT VALUES
% ------------------------------------------------------------

validRows = ...
    ~isnat(data.Date) & ...
    isfinite(data.Sentiment);

removedRows = sum(~validRows);

if removedRows > 0

    fprintf('-> Removing %d invalid rows...\n', ...
        removedRows);

    data = data(validRows,:);

end

if height(data) < 10

    error('step2:insufficientData', ...
        'Too few valid observations remain for classification.');

end

fprintf('   Valid observations: %d\n\n', ...
    height(data));

%% -----------------------------------------------------------
% 8. DEFINE TRAINING PERIOD
% ------------------------------------------------------------
%
% Chronological split:
%
% 70% Training
% 15% Validation
% 15% Test
%
% Classification thresholds are calculated ONLY from training
% sentiment values.

n = height(data);

trainEnd = floor(0.70 * n);

valEnd = floor(0.85 * n);

trainIdx = 1:trainEnd;

valIdx = trainEnd+1:valEnd;

testIdx = valEnd+1:n;

if isempty(trainIdx) || ...
        isempty(valIdx) || ...
        isempty(testIdx)

    error('step2:splitError', ...
        'Unable to create train/validation/test partitions.');

end

trainSentiment = data.Sentiment(trainIdx);

%% -----------------------------------------------------------
% 9. CALCULATE TRAINING-ONLY THRESHOLDS
% ------------------------------------------------------------

trainSentiment = trainSentiment( ...
    isfinite(trainSentiment));

if isempty(trainSentiment)

    error('step2:noTrainingSentiment', ...
        'Training sentiment contains no valid observations.');

end

lowerThreshold = ...
    prctile(trainSentiment, 33.333333);

upperThreshold = ...
    prctile(trainSentiment, 66.666667);

%% -----------------------------------------------------------
% 10. HANDLE IDENTICAL THRESHOLDS
% ------------------------------------------------------------

if lowerThreshold >= upperThreshold

    warning('step2:identicalThresholds', ...
        ['The 33rd and 67th percentile thresholds are identical. ' ...
         'Using standard-deviation-based thresholds instead.']);

    trainMean = mean(trainSentiment, 'omitnan');

    trainStd = std( ...
        trainSentiment, ...
        0, ...
        'omitnan');

    if isfinite(trainStd) && trainStd > 0

        lowerThreshold = ...
            trainMean - 0.5 * trainStd;

        upperThreshold = ...
            trainMean + 0.5 * trainStd;

    else

        lowerThreshold = trainMean;

        upperThreshold = trainMean;

    end

end

%% -----------------------------------------------------------
% 11. CLASSIFY SENTIMENT
% ------------------------------------------------------------

sentimentClass = strings(n,1);

for i = 1:n

    value = data.Sentiment(i);

    if ~isfinite(value)

        sentimentClass(i) = "Unknown";

    elseif value < lowerThreshold

        sentimentClass(i) = "Negative";

    elseif value > upperThreshold

        sentimentClass(i) = "Positive";

    else

        sentimentClass(i) = "Neutral";

    end

end

%% -----------------------------------------------------------
% 12. NUMERIC SENTIMENT REGIME
% ------------------------------------------------------------
%
% -1 = Negative
%  0 = Neutral
% +1 = Positive

sentimentRegime = zeros(n,1);

sentimentRegime(sentimentClass == "Negative") = -1;

sentimentRegime(sentimentClass == "Neutral") = 0;

sentimentRegime(sentimentClass == "Positive") = 1;

%% -----------------------------------------------------------
% 13. ADD RESULTS TO TABLE
% ------------------------------------------------------------

data.SentimentClass = sentimentClass;

data.SentimentRegime = sentimentRegime;

%% -----------------------------------------------------------
% 14. DISPLAY THRESHOLDS
% ------------------------------------------------------------

fprintf('------------------------------------------------------------\n');
fprintf(' [SENTIMENT CLASSIFICATION THRESHOLDS]\n');
fprintf('------------------------------------------------------------\n');

fprintf(' Training observations : %d\n', ...
    numel(trainIdx));

fprintf(' Validation observations: %d\n', ...
    numel(valIdx));

fprintf(' Test observations      : %d\n', ...
    numel(testIdx));

fprintf('\n');

fprintf(' Lower threshold (33%%) : %.6f\n', ...
    lowerThreshold);

fprintf(' Upper threshold (67%%) : %.6f\n', ...
    upperThreshold);

fprintf('------------------------------------------------------------\n\n');

%% -----------------------------------------------------------
% 15. CLASS COUNTS
% ------------------------------------------------------------

negativeCount = sum( ...
    sentimentClass == "Negative");

neutralCount = sum( ...
    sentimentClass == "Neutral");

positiveCount = sum( ...
    sentimentClass == "Positive");

unknownCount = sum( ...
    sentimentClass == "Unknown");

fprintf('------------------------------------------------------------\n');
fprintf(' [SENTIMENT REGIME COUNTS]\n');
fprintf('------------------------------------------------------------\n');

fprintf(' Negative : %d (%.2f%%)\n', ...
    negativeCount, ...
    100 * negativeCount / n);

fprintf(' Neutral  : %d (%.2f%%)\n', ...
    neutralCount, ...
    100 * neutralCount / n);

fprintf(' Positive : %d (%.2f%%)\n', ...
    positiveCount, ...
    100 * positiveCount / n);

if unknownCount > 0

    fprintf(' Unknown  : %d (%.2f%%)\n', ...
        unknownCount, ...
        100 * unknownCount / n);

end

fprintf('------------------------------------------------------------\n\n');

%% -----------------------------------------------------------
% 16. SAVE CLASSIFIED DATASET
% ------------------------------------------------------------

fprintf('-> Saving classified dataset...\n');
fprintf('   Output: %s\n', outputFile);

try

    writetable(data, outputFile);

catch ME

    error('step2:saveError', ...
        'Unable to save classified dataset.\n%s', ...
        ME.message);

end

%% -----------------------------------------------------------
% 17. CREATE SENTIMENT VISUALIZATION
% ------------------------------------------------------------

figureFile = fullfile( ...
    figureDir, ...
    'sentiment_classification.png');

fprintf('-> Generating sentiment classification figure...\n');
fprintf('   Output: %s\n', figureFile);

try

    fig = figure( ...
        'Visible', 'off', ...
        'Position', [100 100 1200 700]);

    plot( ...
        data.Date, ...
        data.Sentiment, ...
        'LineWidth', 1.2);

    hold on;

    yline( ...
        lowerThreshold, ...
        '--', ...
        'Lower Threshold', ...
        'LineWidth', 1.2);

    yline( ...
        upperThreshold, ...
        '--', ...
        'Upper Threshold', ...
        'LineWidth', 1.2);

    xlabel('Date');

    ylabel('Sentiment Score');

    title('BTC Sentiment Classification');

    grid on;

    legend( ...
        'Sentiment', ...
        'Lower Threshold', ...
        'Upper Threshold', ...
        'Location', ...
        'best');

    exportgraphics( ...
        fig, ...
        figureFile, ...
        'Resolution', 150);

    close(fig);

catch ME

    warning('step2:plotError', ...
        ['Unable to create sentiment classification figure.\n' ...
         'Dataset was still saved.\n\n' ...
         'Original error: %s'], ...
        ME.message);

end

%% -----------------------------------------------------------
% 18. FINAL CHECK
% ------------------------------------------------------------

if ~isfile(outputFile)

    error('step2:outputMissing', ...
        'Classified dataset was not created.');

end

fprintf('\n============================================================\n');
fprintf(' [STEP 2 COMPLETE]\n');
fprintf('============================================================\n');

fprintf('Observations          : %d\n', height(data));

fprintf('Negative observations : %d\n', negativeCount);

fprintf('Neutral observations  : %d\n', neutralCount);

fprintf('Positive observations : %d\n', positiveCount);

fprintf('\nClassified dataset:\n%s\n', ...
    outputFile);

fprintf('\nClassification figure:\n%s\n', ...
    figureFile);

fprintf('\nProceeding to Step 3...\n\n');

end