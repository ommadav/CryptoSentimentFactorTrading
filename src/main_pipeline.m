%% ============================================================
% CRYPTO SENTIMENT FACTOR TRADING
% MAIN PIPELINE
% ============================================================
%
% Complete execution pipeline:
%
%   Step 1 -> Data Loading and EDA
%   Step 2 -> Sentiment Classification
%   Step 3 -> Feature Engineering
%   Step 4 -> Model Comparison
%   Step 5 -> Trading Strategy and Backtest
%   Step 6 -> Results Analysis
%
% ============================================================

clc;
clear;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       CRYPTO SENTIMENT FACTOR TRADING PROJECT\n');
fprintf('============================================================\n');
fprintf('\n');


%% ============================================================
% FIND PROJECT DIRECTORY
% ============================================================

thisFile = mfilename('fullpath');

if isempty(thisFile)
    projectDir = pwd;
else
    srcDir = fileparts(thisFile);
    projectDir = fileparts(srcDir);
end


%% ============================================================
% ADD SOURCE DIRECTORIES TO MATLAB PATH
% ============================================================

srcDir = fullfile(projectDir,'src');

utilsDir = fullfile(srcDir,'utils');

if isfolder(srcDir)
    addpath(srcDir);
end

if isfolder(utilsDir)
    addpath(utilsDir);
end


fprintf('Project directory:\n');
fprintf('  %s\n\n',projectDir);


%% ============================================================
% CHECK REQUIRED DIRECTORIES
% ============================================================

dataDir = fullfile(projectDir,'data');

rawDir = fullfile(dataDir,'raw');

processedDir = fullfile(dataDir,'processed');

resultsDir = fullfile(projectDir,'results');

figuresDir = fullfile(resultsDir,'figures');

modelsDir = fullfile(resultsDir,'models');

tablesDir = fullfile(resultsDir,'tables');


requiredDirs = {
    dataDir
    rawDir
    processedDir
    resultsDir
    figuresDir
    modelsDir
    tablesDir
    };


for i = 1:numel(requiredDirs)

    if ~isfolder(requiredDirs{i})

        mkdir(requiredDirs{i});

    end

end


%% ============================================================
% CHECK LIBRARY
% ============================================================

fprintf('Checking project library...\n');

try

    lib = factorTradingLib();

    fprintf('Library loaded successfully.\n\n');

catch ME

    fprintf('\nERROR: factorTradingLib.m could not be loaded.\n');
    fprintf('%s\n',ME.message);

    rethrow(ME);

end


%% ============================================================
% CONFIGURATION
% ============================================================

cfg = lib.config();


fprintf('Configuration:\n');
fprintf('  Prediction horizon : %d day(s)\n', ...
    cfg.predictionHorizon);

fprintf('  Train ratio        : %.0f %%\n', ...
    100*cfg.trainRatio);

fprintf('  Validation ratio   : %.0f %%\n', ...
    100*cfg.validationRatio);

fprintf('  Test ratio         : %.0f %%\n', ...
    100*(1-cfg.trainRatio-cfg.validationRatio));

fprintf('  Initial capital    : %.2f\n', ...
    cfg.initialCapital);

fprintf('\n');


%% ============================================================
% REQUIRED INPUT FILES
% ============================================================

btcFile = fullfile( ...
    rawDir, ...
    'btc_1d_data_2018_to_2025.csv');


sentimentFile = fullfile( ...
    rawDir, ...
    'sentiment_data_consolidated.csv');


fprintf('Checking input files...\n\n');


if ~isfile(btcFile)

    error( ...
        'main_pipeline:MissingBTC', ...
        ['BTC dataset was not found:\n%s\n\n' ...
         'Place the BTC CSV inside data/raw/'], ...
        btcFile);

end


if ~isfile(sentimentFile)

    error( ...
        'main_pipeline:MissingSentiment', ...
        ['Sentiment dataset was not found:\n%s\n\n' ...
         'Place the sentiment CSV inside data/raw/'], ...
        sentimentFile);

end


fprintf('BTC data found:\n');
fprintf('  %s\n\n',btcFile);


fprintf('Sentiment data found:\n');
fprintf('  %s\n\n',sentimentFile);


%% ============================================================
% START TIMER
% ============================================================

totalTimer = tic;


%% ============================================================
% STEP 1
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                       STEP 1\n');
fprintf('              DATA LOADING AND EDA\n');
fprintf('============================================================\n\n');


step1File = ...
    fullfile(srcDir,'step1_data_loading_and_eda.m');


if isfile(step1File)

    try

        run(step1File);

        fprintf('\nSTEP 1 completed successfully.\n');

    catch ME

        fprintf('\nSTEP 1 FAILED.\n');
        fprintf('%s\n',ME.message);

        rethrow(ME);

    end

else

    error( ...
        'main_pipeline:MissingStep1', ...
        'step1_data_loading_and_eda.m was not found.');

end


%% ============================================================
% STEP 2
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                       STEP 2\n');
fprintf('                SENTIMENT CLASSIFICATION\n');
fprintf('============================================================\n\n');


step2File = ...
    fullfile(srcDir,'step2_sentiment_classification.m');


if isfile(step2File)

    try

        run(step2File);

        fprintf('\nSTEP 2 completed successfully.\n');

    catch ME

        fprintf('\nSTEP 2 FAILED.\n');
        fprintf('%s\n',ME.message);

        rethrow(ME);

    end

else

    error( ...
        'main_pipeline:MissingStep2', ...
        'step2_sentiment_classification.m was not found.');

end


%% ============================================================
% STEP 3
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                       STEP 3\n');
fprintf('                  FEATURE ENGINEERING\n');
fprintf('============================================================\n\n');


step3File = ...
    fullfile(srcDir,'step3_feature_engineering.m');


if isfile(step3File)

    try

        run(step3File);

        fprintf('\nSTEP 3 completed successfully.\n');

    catch ME

        fprintf('\nSTEP 3 FAILED.\n');
        fprintf('%s\n',ME.message);

        rethrow(ME);

    end

else

    error( ...
        'main_pipeline:MissingStep3', ...
        'step3_feature_engineering.m was not found.');

end


%% ============================================================
% STEP 4
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                       STEP 4\n');
fprintf('                    MODEL COMPARISON\n');
fprintf('============================================================\n\n');


step4File = ...
    fullfile(srcDir,'step4_model_comparison.m');


if isfile(step4File)

    try

        run(step4File);

        fprintf('\nSTEP 4 completed successfully.\n');

    catch ME

        fprintf('\nSTEP 4 FAILED.\n');
        fprintf('%s\n',ME.message);

        rethrow(ME);

    end

else

    error( ...
        'main_pipeline:MissingStep4', ...
        'step4_model_comparison.m was not found.');

end


%% ============================================================
% STEP 5
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                       STEP 5\n');
fprintf('             TRADING STRATEGY AND BACKTEST\n');
fprintf('============================================================\n\n');


step5File = ...
    fullfile(srcDir,'step5_trading_strategy_backtest.m');


if isfile(step5File)

    try

        run(step5File);

        fprintf('\nSTEP 5 completed successfully.\n');

    catch ME

        fprintf('\nSTEP 5 FAILED.\n');
        fprintf('%s\n',ME.message);

        rethrow(ME);

    end

else

    error( ...
        'main_pipeline:MissingStep5', ...
        'step5_trading_strategy_backtest.m was not found.');

end


%% ============================================================
% STEP 6
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('                       STEP 6\n');
fprintf('                  RESULTS ANALYSIS\n');
fprintf('============================================================\n\n');


step6File = ...
    fullfile(srcDir,'step6_results_analysis.m');


if isfile(step6File)

    try

        run(step6File);

        fprintf('\nSTEP 6 completed successfully.\n');

    catch ME

        fprintf('\nSTEP 6 FAILED.\n');
        fprintf('%s\n',ME.message);

        rethrow(ME);

    end

else

    error( ...
        'main_pipeline:MissingStep6', ...
        'step6_results_analysis.m was not found.');

end


%% ============================================================
% TOTAL EXECUTION TIME
% ============================================================

totalTime = toc(totalTimer);


%% ============================================================
% FINAL PROJECT SUMMARY
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('             COMPLETE PIPELINE FINISHED\n');
fprintf('============================================================\n\n');


fprintf('All six project stages have been executed.\n\n');


fprintf('Project:\n');
fprintf('  Crypto Sentiment Factor Trading\n\n');


fprintf('Pipeline:\n');
fprintf('  1. Data Loading and EDA\n');
fprintf('  2. Sentiment Classification\n');
fprintf('  3. Feature Engineering\n');
fprintf('  4. Model Comparison\n');
fprintf('  5. Trading Strategy and Backtest\n');
fprintf('  6. Results Analysis\n\n');


fprintf('Total execution time:\n');
fprintf('  %.2f seconds\n\n',totalTime);


fprintf('Output directories:\n');
fprintf('  Figures : %s\n',figuresDir);
fprintf('  Models  : %s\n',modelsDir);
fprintf('  Tables  : %s\n',tablesDir);


fprintf('\n');
fprintf('============================================================\n');
fprintf('                    DONE\n');
fprintf('============================================================\n');
fprintf('\n');