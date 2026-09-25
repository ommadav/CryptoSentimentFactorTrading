function pipeline_build_dataset()
%PIPELINE_BUILD_DATASET Build merged dataset from raw price and sentiment CSVs.
% Legacy pipeline script provided for backwards compatibility.

clc;
fprintf('\n============================================================\n');
fprintf(' PIPELINE: BUILD DATASET\n');
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

btcFile = fullfile(projectDir, 'data', 'raw', 'btc_1d_data_2018_to_2025.csv');
sentimentFile = fullfile(projectDir, 'data', 'raw', 'sentiment_data_consolidated.csv');
processedDir = fullfile(projectDir, 'data', 'processed');
if ~exist(processedDir, 'dir')
    mkdir(processedDir);
end

fprintf('Loading BTC price data from:\n  %s\n', btcFile);
fprintf('Loading sentiment data from:\n  %s\n\n', sentimentFile);

merged = lib.loadAndMergeData(btcFile, sentimentFile);

fprintf('Merged (overlap) dataset: %d days\n', height(merged));
fprintf('Date range: %s -> %s\n', ...
    datestr(merged.Date(1), 'dd-mmm-yyyy'), datestr(merged.Date(end), 'dd-mmm-yyyy'));

if height(merged) < 250
    warning('pipeline_build_dataset:smallOverlap', ...
        ['Only %d overlapping days found between the price and sentiment ' ...
         'files. That is a thin dataset for a factor model - treat any ' ...
         'downstream results with proportionate caution.'], height(merged));
end

outputFile = fullfile(processedDir, 'merged_price_sentiment.csv');
writetable(merged, outputFile);

fprintf('\nSaved: %s\n', outputFile);
fprintf('\n============================================================\n');
fprintf(' NEXT: pipeline_train_factor_model.m\n');
fprintf('============================================================\n');

end
