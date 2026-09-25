%% STEP 9B — PREDICTION & THRESHOLD STABILITY DIAGNOSTICS
% Crypto Sentiment Factor Trading
%
% PURPOSE:
%   Diagnose whether model predictions and trading signals remain stable
%   from validation to the untouched test period.
%
% MODELS:
%   1. AR(5)
%   2. Ridge
%   3. Bagged Regression Trees
%   4. Random Forest Regression
%   5. LSBoost Regression
%
% IMPORTANT:
%   - Validation models are trained ONLY on training data.
%   - Step 8 test predictions are reused unchanged.
%   - Step 8 thresholds are reused unchanged.
%   - No test-set tuning is performed.
%
% OUTPUTS:
%   results/tables/STEP9B_PREDICTION_STABILITY.csv
%   results/tables/STEP9B_SIGNAL_STABILITY.csv
%   results/tables/STEP9B_PREDICTION_QUANTILES.csv
%   results/tables/STEP9B_VALIDATION_PREDICTIONS.csv
%   results/tables/STEP9B_THRESHOLD_CROSSINGS.csv
%   results/tables/STEP9B_DIAGNOSTIC_SUMMARY.txt
%
% FIGURES:
%   results/figures/STEP9B_PREDICTION_DISTRIBUTIONS.png
%   results/figures/STEP9B_SIGNAL_COMPARISON.png
%
% -------------------------------------------------------------------------

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' STEP 9B — PREDICTION & THRESHOLD STABILITY DIAGNOSTICS\n');
fprintf('============================================================\n\n');


%% ============================================================
% 1. PROJECT PATHS
% =============================================================

thisFile = mfilename('fullpath');

if isempty(thisFile)

    projectRoot = pwd;

else

    srcFolder = fileparts(thisFile);
    projectRoot = fileparts(srcFolder);

end

dataFolder    = fullfile(projectRoot,'data','processed');
rawFolder     = fullfile(projectRoot,'data','raw');
resultsFolder = fullfile(projectRoot,'results');
tablesFolder  = fullfile(resultsFolder,'tables');
figuresFolder = fullfile(resultsFolder,'figures');
modelsFolder  = fullfile(resultsFolder,'models');

if ~exist(tablesFolder,'dir')
    mkdir(tablesFolder);
end

if ~exist(figuresFolder,'dir')
    mkdir(figuresFolder);
end

if ~exist(modelsFolder,'dir')
    mkdir(modelsFolder);
end


%% ============================================================
% 2. FIXED EXPERIMENTAL SETTINGS
% ============================================================

% These dates match Step 8.

trainStart = datetime(2018,1,6);
trainEnd   = datetime(2020,4,10);

valStart   = datetime(2020,4,11);
valEnd     = datetime(2020,10,4);

testStart  = datetime(2020,10,5);
testEnd    = datetime(2021,3,31);

% EXACT Step 8 thresholds

thresholds.AR5     = 0.0074;
thresholds.Ridge   = 0.0013;
thresholds.Bagged  = 0.0010;
thresholds.RF      = 0.0024;
thresholds.LSBoost = 0.0063;

% Transaction cost
feeRate = 0.001;

% Reproducibility
rng(42);


%% ============================================================
% 3. FILE LOCATIONS
% =============================================================

factorDatasetFile = fullfile( ...
    dataFolder,'factor_dataset.csv');

factorFile = fullfile( ...
    tablesFolder,'STEP7_RIDGE_FACTOR_LIST.csv');

if ~exist(factorFile,'file')

    factorFile = fullfile( ...
        tablesFolder,'STEP8_FACTOR_LIST.csv');

end

predictionFile = fullfile( ...
    tablesFolder,'STEP8_MODEL_PREDICTIONS.csv');


fprintf('Project root:\n%s\n\n',projectRoot);

fprintf('Factor dataset:\n%s\n',factorDatasetFile);

fprintf('Factor list:\n%s\n',factorFile);

fprintf('Step 8 predictions:\n%s\n\n',predictionFile);


if ~exist(factorDatasetFile,'file')

    error('factor_dataset.csv not found.');

end

if ~exist(factorFile,'file')

    error('STEP7_RIDGE_FACTOR_LIST.csv / STEP8_FACTOR_LIST.csv not found.');

end

if ~exist(predictionFile,'file')

    error('STEP8_MODEL_PREDICTIONS.csv not found.');

end


%% ============================================================
% 4. LOAD FACTOR DATASET
% =============================================================

fprintf('Loading factor dataset...\n');

T = readtable( ...
    factorDatasetFile, ...
    'VariableNamingRule','preserve');

fprintf('Rows loaded: %d\n',height(T));
fprintf('Columns loaded: %d\n\n',width(T));


%% ============================================================
% 5. FIND DATE COLUMN
% =============================================================

dateCandidates = { ...
    'Date', ...
    'date', ...
    'time', ...
    'Time', ...
    'Open time', ...
    'OpenTime'};

dateCol = '';

for i = 1:numel(dateCandidates)

    idx = find( ...
        strcmpi(T.Properties.VariableNames, ...
        dateCandidates{i}),1);

    if ~isempty(idx)

        dateCol = T.Properties.VariableNames{idx};

        break;

    end

end

if isempty(dateCol)

    error('Date column not found in factor dataset.');

end


%% ============================================================
% 6. PARSE DATES
% =============================================================

rawDate = T.(dateCol);

if isdatetime(rawDate)

    dates = rawDate;

elseif isnumeric(rawDate)

    try

        dates = datetime( ...
            rawDate,'ConvertFrom','excel');

    catch

        dates = datetime( ...
            rawDate,'ConvertFrom','datenum');

    end

elseif iscell(rawDate) || ...
       isstring(rawDate) || ...
       ischar(rawDate)

    try

        dates = datetime(rawDate);

    catch

        try

            dates = datetime( ...
                rawDate, ...
                'InputFormat','yyyy-MM-dd');

        catch

            error('Could not parse date column.');

        end

    end

else

    error('Unsupported date column type.');

end

dates = dates(:);


%% ============================================================
% 7. SORT CHRONOLOGICALLY
% =============================================================

[dates,sortIdx] = sort(dates);

T = T(sortIdx,:);


%% ============================================================
% 8. LOAD FACTOR LIST
% =============================================================

fprintf('Loading factor list...\n');

F = readtable( ...
    factorFile, ...
    'VariableNamingRule','preserve');

factorNames = string(F{:,1});

factorNames = factorNames( ...
    strlength(strtrim(factorNames)) > 0);

fprintf('Factors listed: %d\n\n', ...
    numel(factorNames));


%% ============================================================
% 9. CHECK WHICH FACTORS EXIST
% =============================================================

availableNames = string( ...
    T.Properties.VariableNames);

keepFactors = strings(0,1);

for i = 1:numel(factorNames)

    if any(strcmp( ...
            availableNames, ...
            factorNames(i)))

        keepFactors(end+1,1) = ...
            factorNames(i); %#ok<SAGROW>

    else

        fprintf( ...
            'WARNING: factor not found: %s\n', ...
            factorNames(i));

    end

end

factorNames = keepFactors;

fprintf('\nFactors available for modeling: %d\n\n', ...
    numel(factorNames));

if numel(factorNames) < 5

    error('Too few usable factors found.');

end


%% ============================================================
% 10. CREATE FACTOR MATRIX
% =============================================================

X = nan(height(T),numel(factorNames));

for j = 1:numel(factorNames)

    colName = factorNames(j);

    idx = find( ...
        strcmp(T.Properties.VariableNames, ...
        colName),1);

    if isempty(idx)
        continue;
    end

    v = T.(T.Properties.VariableNames{idx});

    if isnumeric(v)

        X(:,j) = double(v);

    elseif islogical(v)

        X(:,j) = double(v);

    elseif iscell(v) || ...
           isstring(v) || ...
           ischar(v)

        try

            X(:,j) = str2double(string(v));

        catch

            X(:,j) = NaN(height(T),1);

        end

    end

end


%% ============================================================
% 11. FIND TARGET RETURN
% =============================================================

targetCandidates = { ...
    'TargetReturn1D', ...
    'TargetReturn', ...
    'TargetReturn1d', ...
    'target_return_1d'};

targetCol = '';

for i = 1:numel(targetCandidates)

    idx = find( ...
        strcmpi(T.Properties.VariableNames, ...
        targetCandidates{i}),1);

    if ~isempty(idx)

        targetCol = ...
            T.Properties.VariableNames{idx};

        break;

    end

end


%% ============================================================
% 12. CREATE TARGET IF NECESSARY
% =============================================================

if isempty(targetCol)

    closeIdx = find( ...
        strcmpi(T.Properties.VariableNames,'Close'),1);

    if isempty(closeIdx)

        error( ...
            'Target return and Close column not found.');

    end

    closePrice = double( ...
        T.(T.Properties.VariableNames{closeIdx}));

    y = nan(height(T),1);

    for i = 1:height(T)-1

        if isfinite(closePrice(i)) && ...
           isfinite(closePrice(i+1)) && ...
           closePrice(i) ~= 0

            y(i) = ...
                closePrice(i+1) / ...
                closePrice(i) - 1;

        end

    end

else

    yRaw = T.(targetCol);

    if isnumeric(yRaw)

        y = double(yRaw);

    else

        y = str2double(string(yRaw));

    end

end

y = y(:);


%% ============================================================
% 13. CLEAN DATA
% =============================================================

validRows = ...
    isfinite(y) & ...
    all(isfinite(X),2);

datesValid = dates(validRows);

XValid = X(validRows,:);

yValid = y(validRows);

fprintf('Rows after factor/target cleaning: %d\n\n', ...
    numel(yValid));


%% ============================================================
% 14. FIXED TRAIN / VALIDATION / TEST WINDOWS
% =============================================================

trainMask = ...
    datesValid >= trainStart & ...
    datesValid <= trainEnd;

valMask = ...
    datesValid >= valStart & ...
    datesValid <= valEnd;

testMask = ...
    datesValid >= testStart & ...
    datesValid <= testEnd;


XTrain = XValid(trainMask,:);
yTrain = yValid(trainMask,:);
dTrain = datesValid(trainMask);

XVal = XValid(valMask,:);
yVal = yValid(valMask,:);
dVal = datesValid(valMask);

XTest = XValid(testMask,:);
yTest = yValid(testMask,:);
dTest = datesValid(testMask);


fprintf('Fixed date windows:\n');

fprintf( ...
    'Train:      %s -> %s | %d observations\n', ...
    datestr(trainStart), ...
    datestr(trainEnd), ...
    numel(yTrain));

fprintf( ...
    'Validation: %s -> %s | %d observations\n', ...
    datestr(valStart), ...
    datestr(valEnd), ...
    numel(yVal));

fprintf( ...
    'Test:       %s -> %s | %d observations\n\n', ...
    datestr(testStart), ...
    datestr(testEnd), ...
    numel(yTest));


if isempty(yTrain) || ...
   isempty(yVal) || ...
   isempty(yTest)

    error( ...
        'One or more fixed periods contain no observations.');

end


%% ============================================================
% 15. LOAD STEP 8 TEST PREDICTIONS
% =============================================================

fprintf('Loading Step 8 test predictions...\n');

P8 = readtable( ...
    predictionFile, ...
    'VariableNamingRule','preserve');

fprintf('Step 8 prediction rows: %d\n\n', ...
    height(P8));


%% ============================================================
% 16. DISPLAY ACTUAL STEP 8 COLUMNS
% =============================================================

fprintf('============================================================\n');
fprintf('STEP 8 CSV COLUMNS\n');
fprintf('============================================================\n');

pNames = string(P8.Properties.VariableNames);

for i = 1:numel(pNames)

    fprintf('%2d : %s\n',i,pNames(i));

end

fprintf('\n');


%% ============================================================
% 17. FIND DATE COLUMN IN STEP 8 FILE
% =============================================================

pDateCol = '';

for i = 1:numel(dateCandidates)

    idx = find( ...
        strcmpi(P8.Properties.VariableNames, ...
        dateCandidates{i}),1);

    if ~isempty(idx)

        pDateCol = ...
            P8.Properties.VariableNames{idx};

        break;

    end

end


if isempty(pDateCol)

    idx = find( ...
        strcmpi(P8.Properties.VariableNames, ...
        'TestDate'),1);

    if ~isempty(idx)

        pDateCol = ...
            P8.Properties.VariableNames{idx};

    end

end


if isempty(pDateCol)

    error( ...
        'Date column not found in STEP8_MODEL_PREDICTIONS.csv.');

end


%% ============================================================
% 18. PARSE STEP 8 DATES
% =============================================================

pRawDate = P8.(pDateCol);

if isdatetime(pRawDate)

    pDates = pRawDate;

else

    try

        pDates = datetime(pRawDate);

    catch

        pDates = datetime(string(pRawDate));

    end

end

pDates = pDates(:);


%% ============================================================
% 19. EXTRACT STEP 8 PREDICTION COLUMNS
% =============================================================

fprintf('============================================================\n');
fprintf('IDENTIFYING STEP 8 PREDICTION COLUMNS\n');
fprintf('============================================================\n\n');


predTest.AR5 = ...
    getPredictionColumnFlexible( ...
    P8, ...
    {'AR5_PredictedReturn', ...
     'AR5_Prediction', ...
     'AR5Prediction', ...
     'AR5'});


predTest.Ridge = ...
    getPredictionColumnFlexible( ...
    P8, ...
    {'Ridge_PredictedReturn', ...
     'Ridge_Prediction', ...
     'RidgePrediction', ...
     'Ridge'});


predTest.Bagged = ...
    getPredictionColumnFlexible( ...
    P8, ...
    {'Bagged_PredictedReturn', ...
     'Bagged_Prediction', ...
     'BaggedPrediction', ...
     'Bagged', ...
     'BaggedRegression_PredictedReturn'});


predTest.RF = ...
    getPredictionColumnFlexible( ...
    P8, ...
    {'RandomForest_PredictedReturn', ...
     'RandomForest_Prediction', ...
     'RandomForestPrediction', ...
     'RandomForest', ...
     'RF_PredictedReturn', ...
     'RF_Prediction', ...
     'RFPrediction', ...
     'RF'});


predTest.LSBoost = ...
    getPredictionColumnFlexible( ...
    P8, ...
    {'LSBoost_PredictedReturn', ...
     'LSBoost_Prediction', ...
     'LSBoostPrediction', ...
     'LSBoost'});


fprintf('\nAll five Step 8 prediction columns successfully loaded.\n\n');


%% ============================================================
% 20. ALIGN STEP 8 TEST PREDICTIONS WITH TEST DATA
% =============================================================

[commonDates,ia,ib] = intersect(dTest,pDates);

if numel(commonDates) < ...
        0.90*numel(dTest)

    warning( ...
        ['Only %.1f%% of test dates matched ', ...
         'the Step 8 prediction file.'], ...
        100*numel(commonDates)/numel(dTest));

end


yTestAligned = yTest(ia);

dTestAligned = dTest(ia);


modelNames = { ...
    'AR5', ...
    'Ridge', ...
    'Bagged', ...
    'RF', ...
    'LSBoost'};


for m = 1:numel(modelNames)

    name = modelNames{m};

    predTest.(name) = ...
        predTest.(name)(ib);

end


fprintf( ...
    'Aligned test observations: %d\n\n', ...
    numel(yTestAligned));


%% ============================================================
% 21. BUILD TRUE AR(5) FEATURES
% =============================================================

fprintf('============================================================\n');
fprintf('TRAINING VALIDATION-STAGE MODELS\n');
fprintf('============================================================\n\n');


ARlags = 5;

ARX = nan(numel(yValid),ARlags);

for lag = 1:ARlags

    ARX((lag+1):end,lag) = ...
        yValid(1:end-lag);

end


ARvalid = ...
    all(isfinite(ARX),2) & ...
    isfinite(yValid);

ARdates = datesValid(ARvalid);

ARX = ARX(ARvalid,:);

ARy = yValid(ARvalid);


ARTrainMask = ...
    ARdates >= trainStart & ...
    ARdates <= trainEnd;

ARValMask = ...
    ARdates >= valStart & ...
    ARdates <= valEnd;

ARTestMask = ...
    ARdates >= testStart & ...
    ARdates <= testEnd;


ARXTrain = ARX(ARTrainMask,:);
ARyTrain = ARy(ARTrainMask);

ARXVal = ARX(ARValMask,:);
ARyVal = ARy(ARValMask);

ARXTest = ARX(ARTestMask,:);
ARyTest = ARy(ARTestMask);

ARdVal = ARdates(ARValMask);


%% ============================================================
% 22. AR(5) STANDARDIZATION
% =============================================================

muAR = mean(ARXTrain,1);

sdAR = std(ARXTrain,0,1);

sdAR( ...
    sdAR == 0 | ...
    ~isfinite(sdAR)) = 1;


ARXTrainZ = ...
    (ARXTrain - muAR) ./ sdAR;

ARXValZ = ...
    (ARXVal - muAR) ./ sdAR;


%% ============================================================
% 23. AR(5) MODEL
% =============================================================

ARModel = fitrlinear( ...
    ARXTrainZ, ...
    ARyTrain, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda',1e-4);


predVal.AR5 = ...
    predict(ARModel,ARXValZ);


fprintf('AR5 validation predictions generated.\n');


%% ============================================================
% 24. RIDGE STANDARDIZATION
% =============================================================

ridgeLambda = 31.6227766017;

muRidge = mean(XTrain,1);

sdRidge = std(XTrain,0,1);

sdRidge( ...
    sdRidge == 0 | ...
    ~isfinite(sdRidge)) = 1;


XTrainZ = ...
    (XTrain - muRidge) ./ sdRidge;

XValZ = ...
    (XVal - muRidge) ./ sdRidge;


%% ============================================================
% 25. RIDGE MODEL
% =============================================================

RidgeModel = fitrlinear( ...
    XTrainZ, ...
    yTrain, ...
    'Learner','leastsquares', ...
    'Regularization','ridge', ...
    'Lambda',ridgeLambda);


predVal.Ridge = ...
    predict(RidgeModel,XValZ);


fprintf('Ridge validation predictions generated.\n');


%% ============================================================
% 26. BAGGED REGRESSION MODEL
% =============================================================

fprintf('Training Bagged Regression Trees...\n');


treeTemplateBag = templateTree( ...
    'MinLeafSize',10);


BaggedModel = fitrensemble( ...
    XTrain, ...
    yTrain, ...
    'Method','Bag', ...
    'NumLearningCycles',100, ...
    'Learners',treeTemplateBag);


predVal.Bagged = ...
    predict(BaggedModel,XVal);


fprintf('Bagged validation predictions generated.\n');


%% ============================================================
% 27. RANDOM FOREST MODEL
% =============================================================

fprintf('Training Random Forest Regression...\n');


numFactors = size(XTrain,2);

numPredictorsRF = ...
    round(sqrt(numFactors));


RFModel = TreeBagger( ...
    200, ...
    XTrain, ...
    yTrain, ...
    'Method','regression', ...
    'MinLeafSize',5, ...
    'NumPredictorsToSample',numPredictorsRF, ...
    'OOBPrediction','off');


predVal.RF = ...
    predict(RFModel,XVal);


if iscell(predVal.RF)

    predVal.RF = ...
        str2double(predVal.RF);

end

predVal.RF = ...
    double(predVal.RF(:));


fprintf('Random Forest validation predictions generated.\n');


%% ============================================================
% 28. LSBOOST MODEL
% =============================================================

fprintf('Training LSBoost Regression...\n');


treeTemplateBoost = templateTree( ...
    'MinLeafSize',10);


LSBoostModel = fitrensemble( ...
    XTrain, ...
    yTrain, ...
    'Method','LSBoost', ...
    'NumLearningCycles',100, ...
    'LearnRate',0.05, ...
    'Learners',treeTemplateBoost);


predVal.LSBoost = ...
    predict(LSBoostModel,XVal);


fprintf('LSBoost validation predictions generated.\n\n');


%% ============================================================
% 29. ALIGN VALIDATION DATA
% =============================================================

dValAligned = dVal;

yValAligned = yVal;


%% ============================================================
% 30. CALCULATE PREDICTION STATISTICS
% =============================================================

valStats = struct();

testStats = struct();


for m = 1:numel(modelNames)

    name = modelNames{m};

    valStats.(name) = ...
        predictionStats( ...
        yValAligned, ...
        predVal.(name));

    testStats.(name) = ...
        predictionStats( ...
        yTestAligned, ...
        predTest.(name));

end


%% ============================================================
% 31. PRINT PREDICTION STABILITY RESULTS
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' PREDICTION STABILITY\n');
fprintf('============================================================\n\n');


stabilityRows = table();


for m = 1:numel(modelNames)

    name = modelNames{m};

    V = valStats.(name);

    S = testStats.(name);


    row = table( ...
        string(name), ...
        V.N, ...
        S.N, ...
        V.Mean, ...
        S.Mean, ...
        V.Std, ...
        S.Std, ...
        V.Bias, ...
        S.Bias, ...
        V.RMSE, ...
        S.RMSE, ...
        V.MAE, ...
        S.MAE, ...
        V.Correlation, ...
        S.Correlation, ...
        V.PositivePredictionPct, ...
        S.PositivePredictionPct, ...
        V.NegativePredictionPct, ...
        S.NegativePredictionPct, ...
        'VariableNames',{ ...
        'Model', ...
        'ValidationN', ...
        'TestN', ...
        'ValidationMean', ...
        'TestMean', ...
        'ValidationStd', ...
        'TestStd', ...
        'ValidationBias', ...
        'TestBias', ...
        'ValidationRMSE', ...
        'TestRMSE', ...
        'ValidationMAE', ...
        'TestMAE', ...
        'ValidationCorrelation', ...
        'TestCorrelation', ...
        'ValidationPositivePct', ...
        'TestPositivePct', ...
        'ValidationNegativePct', ...
        'TestNegativePct'});


    stabilityRows = ...
        [stabilityRows;row]; %#ok<AGROW>

end


disp(stabilityRows);


writetable( ...
    stabilityRows, ...
    fullfile( ...
    tablesFolder, ...
    'STEP9B_PREDICTION_STABILITY.csv'));


%% ============================================================
% 32. PREDICTION QUANTILES
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' PREDICTION QUANTILES\n');
fprintf('============================================================\n\n');


quantileRows = table();


for m = 1:numel(modelNames)

    name = modelNames{m};

    V = valStats.(name);

    S = testStats.(name);


    row = table( ...
        string(name), ...
        V.Q05, ...
        V.Q25, ...
        V.Q50, ...
        V.Q75, ...
        V.Q95, ...
        S.Q05, ...
        S.Q25, ...
        S.Q50, ...
        S.Q75, ...
        S.Q95, ...
        'VariableNames',{ ...
        'Model', ...
        'ValidationQ05', ...
        'ValidationQ25', ...
        'ValidationQ50', ...
        'ValidationQ75', ...
        'ValidationQ95', ...
        'TestQ05', ...
        'TestQ25', ...
        'TestQ50', ...
        'TestQ75', ...
        'TestQ95'});


    quantileRows = ...
        [quantileRows;row]; %#ok<AGROW>

end


disp(quantileRows);


writetable( ...
    quantileRows, ...
    fullfile( ...
    tablesFolder, ...
    'STEP9B_PREDICTION_QUANTILES.csv'));


%% ============================================================
% 33. SIGNAL STABILITY
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' SIGNAL STABILITY\n');
fprintf('============================================================\n\n');


signalRows = table();


for m = 1:numel(modelNames)

    name = modelNames{m};

    threshold = thresholds.(name);


    valSignal = ...
        makeSignals( ...
        predVal.(name), ...
        threshold);

    testSignal = ...
        makeSignals( ...
        predTest.(name), ...
        threshold);


    valBUY = sum(valSignal == 1);

    valHOLD = sum(valSignal == 0);

    valSELL = sum(valSignal == -1);


    testBUY = sum(testSignal == 1);

    testHOLD = sum(testSignal == 0);

    testSELL = sum(testSignal == -1);


    row = table( ...
        string(name), ...
        threshold, ...
        valBUY, ...
        valHOLD, ...
        valSELL, ...
        testBUY, ...
        testHOLD, ...
        testSELL, ...
        100*valBUY/numel(valSignal), ...
        100*valHOLD/numel(valSignal), ...
        100*valSELL/numel(valSignal), ...
        100*testBUY/numel(testSignal), ...
        100*testHOLD/numel(testSignal), ...
        100*testSELL/numel(testSignal), ...
        'VariableNames',{ ...
        'Model', ...
        'Threshold', ...
        'ValidationBUY', ...
        'ValidationHOLD', ...
        'ValidationSELL', ...
        'TestBUY', ...
        'TestHOLD', ...
        'TestSELL', ...
        'ValidationBUYPct', ...
        'ValidationHOLDPct', ...
        'ValidationSELLPct', ...
        'TestBUYPct', ...
        'TestHOLDPct', ...
        'TestSELLPct'});


    signalRows = ...
        [signalRows;row]; %#ok<AGROW>


    fprintf('%s | Threshold = %.4f\n', ...
        name,threshold);

    fprintf( ...
        '  Validation: BUY=%d HOLD=%d SELL=%d\n', ...
        valBUY,valHOLD,valSELL);

    fprintf( ...
        '  Test:       BUY=%d HOLD=%d SELL=%d\n\n', ...
        testBUY,testHOLD,testSELL);

end


writetable( ...
    signalRows, ...
    fullfile( ...
    tablesFolder, ...
    'STEP9B_SIGNAL_STABILITY.csv'));


%% ============================================================
% 34. SAVE VALIDATION PREDICTIONS
% =============================================================

validationPredictionTable = table( ...
    dValAligned, ...
    yValAligned, ...
    predVal.AR5, ...
    predVal.Ridge, ...
    predVal.Bagged, ...
    predVal.RF, ...
    predVal.LSBoost, ...
    'VariableNames',{ ...
    'Date', ...
    'ActualReturn', ...
    'AR5_Prediction', ...
    'Ridge_Prediction', ...
    'Bagged_Prediction', ...
    'RF_Prediction', ...
    'LSBoost_Prediction'});


writetable( ...
    validationPredictionTable, ...
    fullfile( ...
    tablesFolder, ...
    'STEP9B_VALIDATION_PREDICTIONS.csv'));


%% ============================================================
% 35. THRESHOLD CROSSING ANALYSIS
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' THRESHOLD CROSSING ANALYSIS\n');
fprintf('============================================================\n\n');


crossingRows = table();


for m = 1:numel(modelNames)

    name = modelNames{m};

    threshold = thresholds.(name);

    v = predVal.(name);

    s = predTest.(name);


    vAbs = abs(v);

    sAbs = abs(s);


    vAbove = ...
        mean(vAbs >= threshold)*100;

    sAbove = ...
        mean(sAbs >= threshold)*100;


    vBuy = ...
        mean(v >= threshold)*100;

    vSell = ...
        mean(v <= -threshold)*100;


    sBuy = ...
        mean(s >= threshold)*100;

    sSell = ...
        mean(s <= -threshold)*100;


    row = table( ...
        string(name), ...
        threshold, ...
        vAbove, ...
        sAbove, ...
        vBuy, ...
        sBuy, ...
        vSell, ...
        sSell, ...
        'VariableNames',{ ...
        'Model', ...
        'Threshold', ...
        'ValidationAbsAboveThresholdPct', ...
        'TestAbsAboveThresholdPct', ...
        'ValidationBUYPct', ...
        'TestBUYPct', ...
        'ValidationSELLPct', ...
        'TestSELLPct'});


    crossingRows = ...
        [crossingRows;row]; %#ok<AGROW>

end


disp(crossingRows);


writetable( ...
    crossingRows, ...
    fullfile( ...
    tablesFolder, ...
    'STEP9B_THRESHOLD_CROSSINGS.csv'));


%% ============================================================
% 36. DIAGNOSTIC SUMMARY FILE
% =============================================================

summaryFile = fullfile( ...
    tablesFolder, ...
    'STEP9B_DIAGNOSTIC_SUMMARY.txt');


fid = fopen(summaryFile,'w');


if fid < 0

    warning( ...
        'Could not create diagnostic summary file.');

else

    fprintf(fid, ...
        'STEP 9B — PREDICTION & THRESHOLD STABILITY DIAGNOSTICS\n');

    fprintf(fid, ...
        '============================================================\n\n');


    fprintf(fid,'Purpose:\n');

    fprintf(fid, ...
        ['Compare model prediction distributions and ', ...
         'trading signals between validation and ', ...
         'the untouched test period.\n\n']);


    fprintf(fid,'Data periods:\n');

    fprintf(fid, ...
        'Train:      %s -> %s\n', ...
        datestr(trainStart), ...
        datestr(trainEnd));

    fprintf(fid, ...
        'Validation: %s -> %s\n', ...
        datestr(valStart), ...
        datestr(valEnd));

    fprintf(fid, ...
        'Test:       %s -> %s\n\n', ...
        datestr(testStart), ...
        datestr(testEnd));


    fprintf(fid,'Methodology:\n');

    fprintf(fid, ...
        '- Validation models are trained only on training data.\n');

    fprintf(fid, ...
        '- Step 8 test predictions are reused unchanged.\n');

    fprintf(fid, ...
        '- Step 8 thresholds are reused unchanged.\n');

    fprintf(fid, ...
        '- No test-set tuning is performed.\n\n');


    fprintf(fid,'Prediction statistics:\n\n');


    for m = 1:numel(modelNames)

        name = modelNames{m};

        V = valStats.(name);

        S = testStats.(name);


        fprintf(fid,'MODEL: %s\n',name);

        fprintf(fid, ...
            '  Validation mean prediction: %.8f\n', ...
            V.Mean);

        fprintf(fid, ...
            '  Test mean prediction:       %.8f\n', ...
            S.Mean);


        fprintf(fid, ...
            '  Validation std prediction:  %.8f\n', ...
            V.Std);

        fprintf(fid, ...
            '  Test std prediction:        %.8f\n', ...
            S.Std);


        fprintf(fid, ...
            '  Validation bias:            %.8f\n', ...
            V.Bias);

        fprintf(fid, ...
            '  Test bias:                  %.8f\n', ...
            S.Bias);


        fprintf(fid, ...
            '  Validation RMSE:            %.8f\n', ...
            V.RMSE);

        fprintf(fid, ...
            '  Test RMSE:                  %.8f\n', ...
            S.RMSE);


        fprintf(fid, ...
            '  Validation correlation:     %.8f\n', ...
            V.Correlation);

        fprintf(fid, ...
            '  Test correlation:           %.8f\n\n', ...
            S.Correlation);

    end


    fprintf(fid,'Signal stability:\n\n');


    for m = 1:numel(modelNames)

        name = modelNames{m};

        threshold = thresholds.(name);


        valSignal = ...
            makeSignals( ...
            predVal.(name), ...
            threshold);

        testSignal = ...
            makeSignals( ...
            predTest.(name), ...
            threshold);


        fprintf(fid,'%s\n',name);

        fprintf(fid, ...
            '  Threshold: %.4f\n', ...
            threshold);


        fprintf(fid, ...
            '  Validation BUY/HOLD/SELL: %d / %d / %d\n', ...
            sum(valSignal == 1), ...
            sum(valSignal == 0), ...
            sum(valSignal == -1));


        fprintf(fid, ...
            '  Test BUY/HOLD/SELL:       %d / %d / %d\n\n', ...
            sum(testSignal == 1), ...
            sum(testSignal == 0), ...
            sum(testSignal == -1));

    end


    fclose(fid);

end


%% ============================================================
% 37. PREDICTION DISTRIBUTION FIGURE
% =============================================================

figure( ...
    'Color','w', ...
    'Position',[100 100 1200 700]);


hold on;


legendEntries = {};


for m = 1:numel(modelNames)

    name = modelNames{m};


    histogram( ...
        predVal.(name), ...
        25, ...
        'Normalization','probability', ...
        'DisplayStyle','stairs', ...
        'LineWidth',1.5);


    legendEntries{end+1} = ...
        [name ' Validation']; %#ok<SAGROW>


    histogram( ...
        predTest.(name), ...
        25, ...
        'Normalization','probability', ...
        'DisplayStyle','stairs', ...
        'LineWidth',1.5);


    legendEntries{end+1} = ...
        [name ' Test']; %#ok<SAGROW>

end


xlabel('Predicted 1-Day Return');

ylabel('Probability');

title( ...
    'Validation vs Test Prediction Distributions');


legend( ...
    legendEntries, ...
    'Location','bestoutside');


grid on;

box on;


saveas( ...
    gcf, ...
    fullfile( ...
    figuresFolder, ...
    'STEP9B_PREDICTION_DISTRIBUTIONS.png'));


%% ============================================================
% 38. SIGNAL COMPARISON FIGURE
% =============================================================

figure( ...
    'Color','w', ...
    'Position',[100 100 1200 700]);


signalMatrixVal = ...
    zeros(numel(modelNames),3);

signalMatrixTest = ...
    zeros(numel(modelNames),3);


for m = 1:numel(modelNames)

    name = modelNames{m};

    threshold = thresholds.(name);


    sV = makeSignals( ...
        predVal.(name), ...
        threshold);

    sT = makeSignals( ...
        predTest.(name), ...
        threshold);


    signalMatrixVal(m,:) = [ ...
        sum(sV == 1), ...
        sum(sV == 0), ...
        sum(sV == -1)];


    signalMatrixTest(m,:) = [ ...
        sum(sT == 1), ...
        sum(sT == 0), ...
        sum(sT == -1)];

end


subplot(1,2,1);


bar(signalMatrixVal,'grouped');


set(gca, ...
    'XTick',1:numel(modelNames), ...
    'XTickLabel',modelNames);


xlabel('Model');

ylabel('Number of Observations');

title('Validation Signals');


legend( ...
    {'BUY','HOLD','SELL'}, ...
    'Location','best');


grid on;


subplot(1,2,2);


bar(signalMatrixTest,'grouped');


set(gca, ...
    'XTick',1:numel(modelNames), ...
    'XTickLabel',modelNames);


xlabel('Model');

ylabel('Number of Observations');

title('Test Signals');


legend( ...
    {'BUY','HOLD','SELL'}, ...
    'Location','best');


grid on;


saveas( ...
    gcf, ...
    fullfile( ...
    figuresFolder, ...
    'STEP9B_SIGNAL_COMPARISON.png'));


%% ============================================================
% 39. FINAL CONSOLE SUMMARY
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' STEP 9B COMPLETE\n');
fprintf('============================================================\n\n');


fprintf('Files created:\n\n');

fprintf( ...
    '1. STEP9B_PREDICTION_STABILITY.csv\n');

fprintf( ...
    '2. STEP9B_SIGNAL_STABILITY.csv\n');

fprintf( ...
    '3. STEP9B_PREDICTION_QUANTILES.csv\n');

fprintf( ...
    '4. STEP9B_VALIDATION_PREDICTIONS.csv\n');

fprintf( ...
    '5. STEP9B_THRESHOLD_CROSSINGS.csv\n');

fprintf( ...
    '6. STEP9B_DIAGNOSTIC_SUMMARY.txt\n');

fprintf( ...
    '7. STEP9B_PREDICTION_DISTRIBUTIONS.png\n');

fprintf( ...
    '8. STEP9B_SIGNAL_COMPARISON.png\n\n');


fprintf('============================================================\n');
fprintf(' INTERPRETATION GUIDE\n');
fprintf('============================================================\n\n');


fprintf( ...
    ['1. Large change in prediction mean/std between ', ...
     'validation and test indicates prediction distribution shift.\n\n']);


fprintf( ...
    ['2. Large change in BUY/HOLD/SELL counts indicates ', ...
     'threshold or regime sensitivity.\n\n']);


fprintf( ...
    ['3. Large increase in SELL signals means the model ', ...
     'produced substantially more negative predictions during test.\n\n']);


fprintf( ...
    ['4. Prediction correlation near zero indicates weak ', ...
     'linear relationship between predicted and actual returns.\n\n']);


fprintf( ...
    ['5. Large increase in test RMSE indicates weaker ', ...
     'out-of-sample predictive performance.\n\n']);


fprintf( ...
    ['6. These diagnostics are descriptive and do not use ', ...
     'the test set for tuning.\n\n']);


fprintf('Next research step after Step 9B:\n');

fprintf( ...
    'STEP 10 — Reduced-Factor / Multicollinearity Experiment\n\n');


fprintf('Done.\n');


%% ============================================================
% LOCAL FUNCTIONS
% =============================================================


function pred = getPredictionColumnFlexible(P,candidateNames)

    names = string(P.Properties.VariableNames);

    foundIdx = [];


    % ---------------------------------------------------------
    % METHOD 1: EXACT CASE-INSENSITIVE MATCH
    % ---------------------------------------------------------

    for i = 1:numel(candidateNames)

        hit = find( ...
            strcmpi(names,candidateNames{i}),1);

        if ~isempty(hit)

            foundIdx = hit;

            break;

        end

    end


    % ---------------------------------------------------------
    % METHOD 2: NORMALIZED MATCH
    % Removes:
    %   _
    %   spaces
    %   -
    % ---------------------------------------------------------

    if isempty(foundIdx)

        normalizedColumns = lower(names);

        normalizedColumns = ...
            replace(normalizedColumns,"_","");

        normalizedColumns = ...
            replace(normalizedColumns," ","");

        normalizedColumns = ...
            replace(normalizedColumns,"-","");


        for i = 1:numel(candidateNames)

            candidate = ...
                lower(string(candidateNames{i}));

            candidate = ...
                replace(candidate,"_","");

            candidate = ...
                replace(candidate," ","");

            candidate = ...
                replace(candidate,"-","");


            hit = find( ...
                strcmp(normalizedColumns,candidate),1);


            if ~isempty(hit)

                foundIdx = hit;

                break;

            end

        end

    end


    % ---------------------------------------------------------
    % METHOD 3: SEARCH FOR MODEL + PREDICTION
    % ---------------------------------------------------------

    if isempty(foundIdx)

        for i = 1:numel(candidateNames)

            candidate = ...
                lower(string(candidateNames{i}));

            candidate = ...
                replace(candidate,"_","");

            candidate = ...
                replace(candidate," ","");

            candidate = ...
                replace(candidate,"-","");


            for j = 1:numel(names)

                col = lower(names(j));

                col = ...
                    replace(col,"_","");

                col = ...
                    replace(col," ","");

                col = ...
                    replace(col,"-","");


                if contains(col,candidate) && ...
                        contains(col,"pred")

                    foundIdx = j;

                    break;

                end

            end


            if ~isempty(foundIdx)

                break;

            end

        end

    end


    % ---------------------------------------------------------
    % METHOD 4: MODEL-SPECIFIC FALLBACKS
    % ---------------------------------------------------------

    if isempty(foundIdx)

        lowerNames = lower(names);


        % Random Forest
        if any(strcmpi(candidateNames, ...
                'RandomForest'))

            hit = find( ...
                contains(lowerNames,'randomforest') & ...
                contains(lowerNames,'return'),1);

            if isempty(hit)

                hit = find( ...
                    contains(lowerNames,'randomforest') & ...
                    contains(lowerNames,'pred'),1);

            end

            if ~isempty(hit)

                foundIdx = hit;

            end

        end

    end


    % ---------------------------------------------------------
    % FAILURE: PRINT ALL COLUMNS
    % ---------------------------------------------------------

    if isempty(foundIdx)

        fprintf('\n');
        fprintf('Available Step 8 columns:\n');

        for j = 1:numel(names)

            fprintf( ...
                '  %2d: %s\n', ...
                j,names(j));

        end

        fprintf('\n');

        error( ...
            ['Could not identify prediction column. ', ...
             'Check STEP8_MODEL_PREDICTIONS.csv.']);

    end


    selectedColumn = ...
        P.Properties.VariableNames{foundIdx};


    fprintf( ...
        'Using prediction column: %s\n', ...
        selectedColumn);


    v = P.(selectedColumn);


    if isnumeric(v)

        pred = double(v);

    elseif iscell(v) || ...
           isstring(v) || ...
           ischar(v)

        pred = ...
            str2double(string(v));

    else

        error( ...
            'Prediction column has unsupported data type.');

    end


    pred = pred(:);

end



function stats = predictionStats(actual,pred)

    actual = actual(:);

    pred = pred(:);


    valid = ...
        isfinite(actual) & ...
        isfinite(pred);


    actual = actual(valid);

    pred = pred(valid);


    stats.N = numel(actual);


    stats.Mean = mean(pred);

    stats.Std = std(pred);


    stats.ActualMean = mean(actual);

    stats.ActualStd = std(actual);


    stats.Bias = ...
        mean(pred - actual);


    stats.RMSE = ...
        sqrt(mean((pred - actual).^2));


    stats.MAE = ...
        mean(abs(pred - actual));


    if std(pred) > 0 && ...
       std(actual) > 0

        stats.Correlation = ...
            corr(pred,actual);

    else

        stats.Correlation = NaN;

    end


    stats.PositivePredictionPct = ...
        100*mean(pred > 0);


    stats.NegativePredictionPct = ...
        100*mean(pred < 0);


    stats.ZeroPredictionPct = ...
        100*mean(pred == 0);


    stats.Q05 = prctile(pred,5);

    stats.Q25 = prctile(pred,25);

    stats.Q50 = prctile(pred,50);

    stats.Q75 = prctile(pred,75);

    stats.Q95 = prctile(pred,95);

end



function signal = makeSignals(pred,threshold)

    signal = zeros(size(pred));


    signal(pred >= threshold) = 1;

    signal(pred <= -threshold) = -1;

end