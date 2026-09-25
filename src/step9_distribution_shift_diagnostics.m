%% STEP 9 - DISTRIBUTION SHIFT & PREDICTION DIAGNOSTICS
% Crypto Sentiment Factor Trading
%
% Purpose:
%   Diagnose why the multivariate factor models from Step 8
%   performed differently between validation and test.
%
% This script DOES NOT:
%   - retrain models
%   - change thresholds
%   - select a better model
%   - optimize using the test set
%
% It only performs diagnostics on:
%   1. Factor distribution shift
%   2. Prediction distribution shift
%   3. Actual-return distribution shift
%   4. Sentiment regime shift
%   5. Factor correlation / multicollinearity
%   6. Prediction bias
%   7. Signal distribution
%   8. Validation vs test trading behavior
%
% Inputs:
%   factor_dataset.csv
%   STEP8_MODEL_PREDICTIONS.csv
%   STEP7_RIDGE_FACTOR_LIST.csv
%
% Outputs:
%   STEP9_DISTRIBUTION_SHIFT_SUMMARY.csv
%   STEP9_FACTOR_SHIFT.csv
%   STEP9_PREDICTION_SHIFT.csv
%   STEP9_CORRELATION_MATRIX.csv
%   STEP9_SIGNAL_DISTRIBUTION.csv
%   STEP9_DIAGNOSTIC_SUMMARY.txt
%
% Figures:
%   STEP9_FACTOR_SHIFT.png
%   STEP9_PREDICTION_DISTRIBUTIONS.png
%   STEP9_ACTUAL_RETURN_DISTRIBUTION.png
%   STEP9_SENTIMENT_SHIFT.png
%   STEP9_FACTOR_CORRELATION.png
%   STEP9_PREDICTION_VS_ACTUAL.png
%   STEP9_SIGNAL_DISTRIBUTION.png
%
% Research/backtesting diagnostics only.

clear;
clc;
close all;

fprintf('\n');
fprintf('============================================================\n');
fprintf(' STEP 9 - DISTRIBUTION SHIFT & PREDICTION DIAGNOSTICS\n');
fprintf('============================================================\n');

%% ============================================================
% 1. PATH SETUP
% =============================================================

thisFile = mfilename('fullpath');

if isempty(thisFile)
    projectRoot = pwd;
else
    srcFolder = fileparts(thisFile);
    projectRoot = fileparts(srcFolder);
end

dataFile = fullfile( ...
    projectRoot, ...
    'data', ...
    'processed', ...
    'factor_dataset.csv');

if ~isfile(dataFile)

    dataFile = fullfile( ...
        projectRoot, ...
        'data', ...
        'raw', ...
        'factor_dataset.csv');

end

if ~isfile(dataFile)
    error('factor_dataset.csv not found.');
end

tableDir = fullfile(projectRoot,'results','tables');
figureDir = fullfile(projectRoot,'results','figures');

if ~exist(tableDir,'dir')
    mkdir(tableDir);
end

if ~exist(figureDir,'dir')
    mkdir(figureDir);
end

step8PredictionFile = fullfile( ...
    tableDir, ...
    'STEP8_MODEL_PREDICTIONS.csv');

step7FactorFile = fullfile( ...
    tableDir, ...
    'STEP7_RIDGE_FACTOR_LIST.csv');

if ~isfile(step8PredictionFile)
    error('STEP8_MODEL_PREDICTIONS.csv not found.');
end

if ~isfile(step7FactorFile)
    error('STEP7_RIDGE_FACTOR_LIST.csv not found.');
end

fprintf('\nProject root:\n%s\n',projectRoot);

%% ============================================================
% 2. FIXED PERIODS
% =============================================================

trainStart = datetime(2018,1,6);
trainEnd   = datetime(2020,4,10);

valStart   = datetime(2020,4,11);
valEnd     = datetime(2020,10,4);

testStart  = datetime(2020,10,5);
testEnd    = datetime(2021,3,31);

fprintf('\n');
fprintf('Validation : %s -> %s\n', ...
    datestr(valStart,'yyyy-mm-dd'), ...
    datestr(valEnd,'yyyy-mm-dd'));

fprintf('Test       : %s -> %s\n', ...
    datestr(testStart,'yyyy-mm-dd'), ...
    datestr(testEnd,'yyyy-mm-dd'));

%% ============================================================
% 3. LOAD DATA
% =============================================================

fprintf('\n============================================================\n');
fprintf('LOADING DATA\n');
fprintf('============================================================\n');

T = readtable(dataFile,'VariableNamingRule','preserve');

fprintf('Rows    : %d\n',height(T));
fprintf('Columns : %d\n',width(T));

%% ============================================================
% 4. DATE PARSING
% =============================================================

dateCandidates = { ...
    'Date', ...
    'date', ...
    'Open time', ...
    'time', ...
    'Timestamp'};

dateName = '';

for i = 1:numel(dateCandidates)

    idx = find( ...
        strcmpi(string(T.Properties.VariableNames), ...
        dateCandidates{i}),1);

    if ~isempty(idx)

        dateName = T.Properties.VariableNames{idx};
        break;

    end

end

if isempty(dateName)
    error('Date column not found.');
end

rawDate = T.(dateName);

if isdatetime(rawDate)

    dates = rawDate;

elseif isnumeric(rawDate)

    try
        dates = datetime(rawDate,'ConvertFrom','excel');
    catch
        dates = datetime(rawDate,'ConvertFrom','posixtime');
    end

else

    rawString = string(rawDate);

    dates = NaT(size(rawString));

    formats = { ...
        'yyyy-MM-dd', ...
        'yyyy-MM-dd HH:mm:ss', ...
        'yyyy-MM-dd HH:mm:ss.SSS', ...
        'dd-MM-yyyy', ...
        'MM/dd/yyyy', ...
        'yyyy/MM/dd'};

    for f = 1:numel(formats)

        missing = isnat(dates);

        if ~any(missing)
            break;
        end

        try
            dates(missing) = datetime( ...
                rawString(missing), ...
                'InputFormat',formats{f});
        catch
        end

    end

    missing = isnat(dates);

    if any(missing)

        try
            dates(missing) = datetime(rawString(missing));
        catch
        end

    end

end

T.Date = dates;

validDate = ~isnat(T.Date);

T = T(validDate,:);

T = sortrows(T,'Date');

[~,uniqueIdx] = unique(T.Date,'stable');

T = T(uniqueIdx,:);

%% ============================================================
% 5. LOAD EXACT STEP-7 FACTOR LIST
% =============================================================

F = readtable( ...
    step7FactorFile, ...
    'VariableNamingRule','preserve');

factorNames = cellstr(string(F{:,1}));

factorNames = factorNames( ...
    ismember(factorNames,T.Properties.VariableNames));

factorNames = unique(factorNames,'stable');

fprintf('\nNumber of factors: %d\n',numel(factorNames));

%% ============================================================
% 6. CREATE FACTOR MATRIX
% =============================================================

X = NaN(height(T),numel(factorNames));

for j = 1:numel(factorNames)

    x = T.(factorNames{j});

    if iscell(x)

        x = str2double(string(x));

    elseif isstring(x) || ischar(x) || iscategorical(x)

        x = str2double(string(x));

    end

    X(:,j) = double(x);

end

%% ============================================================
% 7. COMMON DATE WINDOWS
% =============================================================

idxTrain = T.Date >= trainStart & T.Date <= trainEnd;
idxVal   = T.Date >= valStart   & T.Date <= valEnd;
idxTest  = T.Date >= testStart  & T.Date <= testEnd;

fprintf('\nDate-window observations:\n');
fprintf('Train : %d\n',sum(idxTrain));
fprintf('Val   : %d\n',sum(idxVal));
fprintf('Test  : %d\n',sum(idxTest));

%% ============================================================
% 8. FACTOR DISTRIBUTION SHIFT
% =============================================================

fprintf('\n============================================================\n');
fprintf('FACTOR DISTRIBUTION SHIFT\n');
fprintf('============================================================\n');

nFactors = numel(factorNames);

factorMeanTrain = NaN(nFactors,1);
factorMeanVal = NaN(nFactors,1);
factorMeanTest = NaN(nFactors,1);

factorStdTrain = NaN(nFactors,1);
factorStdVal = NaN(nFactors,1);
factorStdTest = NaN(nFactors,1);

factorMedianTrain = NaN(nFactors,1);
factorMedianVal = NaN(nFactors,1);
factorMedianTest = NaN(nFactors,1);

factorMinTrain = NaN(nFactors,1);
factorMaxTrain = NaN(nFactors,1);

factorMinVal = NaN(nFactors,1);
factorMaxVal = NaN(nFactors,1);

factorMinTest = NaN(nFactors,1);
factorMaxTest = NaN(nFactors,1);

meanShiftValZ = NaN(nFactors,1);
meanShiftTestZ = NaN(nFactors,1);

stdRatioTestTrain = NaN(nFactors,1);

for j = 1:nFactors

    x = X(:,j);

    a = x(idxTrain);
    b = x(idxVal);
    c = x(idxTest);

    a = a(isfinite(a));
    b = b(isfinite(b));
    c = c(isfinite(c));

    if isempty(a) || isempty(b) || isempty(c)
        continue;
    end

    factorMeanTrain(j) = mean(a);
    factorMeanVal(j) = mean(b);
    factorMeanTest(j) = mean(c);

    factorStdTrain(j) = std(a);
    factorStdVal(j) = std(b);
    factorStdTest(j) = std(c);

    factorMedianTrain(j) = median(a);
    factorMedianVal(j) = median(b);
    factorMedianTest(j) = median(c);

    factorMinTrain(j) = min(a);
    factorMaxTrain(j) = max(a);

    factorMinVal(j) = min(b);
    factorMaxVal(j) = max(b);

    factorMinTest(j) = min(c);
    factorMaxTest(j) = max(c);

    if factorStdTrain(j) > 0

        meanShiftValZ(j) = ...
            (factorMeanVal(j)-factorMeanTrain(j)) / ...
            factorStdTrain(j);

        meanShiftTestZ(j) = ...
            (factorMeanTest(j)-factorMeanTrain(j)) / ...
            factorStdTrain(j);

    end

    if factorStdTrain(j) > 0

        stdRatioTestTrain(j) = ...
            factorStdTest(j)/factorStdTrain(j);

    end

end

factorShiftTable = table( ...
    string(factorNames(:)), ...
    factorMeanTrain, ...
    factorMeanVal, ...
    factorMeanTest, ...
    factorStdTrain, ...
    factorStdVal, ...
    factorStdTest, ...
    factorMedianTrain, ...
    factorMedianVal, ...
    factorMedianTest, ...
    meanShiftValZ, ...
    meanShiftTestZ, ...
    stdRatioTestTrain, ...
    factorMinTrain, ...
    factorMaxTrain, ...
    factorMinVal, ...
    factorMaxVal, ...
    factorMinTest, ...
    factorMaxTest, ...
    'VariableNames',{ ...
    'Factor', ...
    'TrainMean', ...
    'ValidationMean', ...
    'TestMean', ...
    'TrainStd', ...
    'ValidationStd', ...
    'TestStd', ...
    'TrainMedian', ...
    'ValidationMedian', ...
    'TestMedian', ...
    'ValidationMeanShiftZ', ...
    'TestMeanShiftZ', ...
    'TestToTrainStdRatio', ...
    'TrainMin', ...
    'TrainMax', ...
    'ValidationMin', ...
    'ValidationMax', ...
    'TestMin', ...
    'TestMax'});

writetable( ...
    factorShiftTable, ...
    fullfile(tableDir,'STEP9_FACTOR_SHIFT.csv'));

%% ============================================================
% 9. IDENTIFY LARGEST FACTOR SHIFTS
% =============================================================

absShift = abs(meanShiftTestZ);

[sortedShift,sortIdx] = sort(absShift,'descend','MissingPlacement','last');

fprintf('\nTop factor mean shifts: validation -> test\n');
fprintf('------------------------------------------------------------\n');

topN = min(10,nFactors);

for i = 1:topN

    j = sortIdx(i);

    fprintf('%2d. %-30s Z-shift = %+8.3f\n', ...
        i, ...
        factorNames{j}, ...
        meanShiftTestZ(j));

end

%% ============================================================
% 10. SENTIMENT DIAGNOSTICS
% =============================================================

fprintf('\n============================================================\n');
fprintf('SENTIMENT REGIME DIAGNOSTICS\n');
fprintf('============================================================\n');

sentimentName = '';

sentimentCandidates = { ...
    'Sentiment', ...
    'sentiment'};

for i = 1:numel(sentimentCandidates)

    idx = find( ...
        strcmpi(string(T.Properties.VariableNames), ...
        sentimentCandidates{i}),1);

    if ~isempty(idx)

        sentimentName = T.Properties.VariableNames{idx};
        break;

    end

end

if ~isempty(sentimentName)

    sentiment = double(T.(sentimentName));

    trainSent = sentiment(idxTrain);
    valSent = sentiment(idxVal);
    testSent = sentiment(idxTest);

    trainSent = trainSent(isfinite(trainSent));
    valSent = valSent(isfinite(valSent));
    testSent = testSent(isfinite(testSent));

    fprintf('Train sentiment mean : %.6f\n',mean(trainSent));
    fprintf('Validation mean      : %.6f\n',mean(valSent));
    fprintf('Test mean            : %.6f\n',mean(testSent));

    fprintf('Train sentiment std  : %.6f\n',std(trainSent));
    fprintf('Validation std       : %.6f\n',std(valSent));
    fprintf('Test std             : %.6f\n',std(testSent));

    sentimentTable = table( ...
        ["Train";"Validation";"Test"], ...
        [mean(trainSent);mean(valSent);mean(testSent)], ...
        [std(trainSent);std(valSent);std(testSent)], ...
        [median(trainSent);median(valSent);median(testSent)], ...
        'VariableNames',{ ...
        'Period', ...
        'Mean', ...
        'Std', ...
        'Median'});

    writetable( ...
        sentimentTable, ...
        fullfile(tableDir,'STEP9_SENTIMENT_SHIFT.csv'));

else

    fprintf('Sentiment column not found.\n');

end

%% ============================================================
% 11. LOAD STEP 8 PREDICTIONS
% =============================================================

fprintf('\n============================================================\n');
fprintf('LOADING STEP 8 PREDICTIONS\n');
fprintf('============================================================\n');

P = readtable( ...
    step8PredictionFile, ...
    'VariableNamingRule','preserve');

fprintf('Prediction rows: %d\n',height(P));

%% ============================================================
% 12. PARSE PREDICTION DATE
% =============================================================

if isdatetime(P.Date)

    predictionDates = P.Date;

else

    try

        predictionDates = datetime( ...
            P.Date, ...
            'InputFormat','yyyy-MM-dd');

    catch

        predictionDates = datetime(P.Date);

    end

end

P.Date = predictionDates;

%% ============================================================
% 13. PREDICTION DISTRIBUTION DIAGNOSTICS
% =============================================================

fprintf('\n============================================================\n');
fprintf('PREDICTION DISTRIBUTION SHIFT\n');
fprintf('============================================================\n');

modelPrefixes = { ...
    'AR5', ...
    'Ridge', ...
    'BaggedRegression', ...
    'RandomForest', ...
    'LSBoost'};

nModels = numel(modelPrefixes);

predShiftRows = [];

predictionShiftModel = strings(nModels,1);

predMeanVal = NaN(nModels,1);
predMeanTest = NaN(nModels,1);

predStdVal = NaN(nModels,1);
predStdTest = NaN(nModels,1);

predMinVal = NaN(nModels,1);
predMaxVal = NaN(nModels,1);

predMinTest = NaN(nModels,1);
predMaxTest = NaN(nModels,1);

predMeanShiftZ = NaN(nModels,1);

actualMeanVal = NaN(nModels,1);
actualMeanTest = NaN(nModels,1);

actualStdVal = NaN(nModels,1);
actualStdTest = NaN(nModels,1);

predictionBiasVal = NaN(nModels,1);
predictionBiasTest = NaN(nModels,1);

correlationVal = NaN(nModels,1);
correlationTest = NaN(nModels,1);

for m = 1:nModels

    model = modelPrefixes{m};

    predVar = [model '_PredictedReturn'];
    actualVar = [model '_ActualReturn'];

    if ~ismember(predVar,P.Properties.VariableNames)

        fprintf('%s prediction column not found.\n',model);
        continue;

    end

    pred = double(P.(predVar));

    if ismember(actualVar,P.Properties.VariableNames)
        actual = double(P.(actualVar));
    else
        actual = double(P.AR5_ActualReturn);
    end

    valMask = P.Date >= valStart & P.Date <= valEnd;
    testMask = P.Date >= testStart & P.Date <= testEnd;

    pv = pred(valMask);
    pt = pred(testMask);

    av = actual(valMask);
    at = actual(testMask);

    pv = pv(isfinite(pv));
    pt = pt(isfinite(pt));

    av = av(isfinite(av));
    at = at(isfinite(at));

    predictionShiftModel(m) = string(model);

    if ~isempty(pv) && ~isempty(pt)

        predMeanVal(m) = mean(pv);
        predMeanTest(m) = mean(pt);

        predStdVal(m) = std(pv);
        predStdTest(m) = std(pt);

        predMinVal(m) = min(pv);
        predMaxVal(m) = max(pv);

        predMinTest(m) = min(pt);
        predMaxTest(m) = max(pt);

        if predStdVal(m) > 0

            predMeanShiftZ(m) = ...
                (predMeanTest(m)-predMeanVal(m))/predStdVal(m);

        end

    end

    if ~isempty(av) && ~isempty(at)

        actualMeanVal(m) = mean(av);
        actualMeanTest(m) = mean(at);

        actualStdVal(m) = std(av);
        actualStdTest(m) = std(at);

    end

    % Prediction bias
    %
    % Positive bias = model predicts too high on average.
    % Negative bias = model predicts too low on average.

    if ~isempty(pv) && ~isempty(av)

        n = min(numel(pv),numel(av));

        predictionBiasVal(m) = ...
            mean(pv(1:n)-av(1:n));

        correlationVal(m) = ...
            safeCorrelation(av(1:n),pv(1:n));

    end

    if ~isempty(pt) && ~isempty(at)

        n = min(numel(pt),numel(at));

        predictionBiasTest(m) = ...
            mean(pt(1:n)-at(1:n));

        correlationTest(m) = ...
            safeCorrelation(at(1:n),pt(1:n));

    end

end

predictionShiftTable = table( ...
    predictionShiftModel, ...
    predMeanVal, ...
    predMeanTest, ...
    predStdVal, ...
    predStdTest, ...
    predMinVal, ...
    predMaxVal, ...
    predMinTest, ...
    predMaxTest, ...
    predMeanShiftZ, ...
    actualMeanVal, ...
    actualMeanTest, ...
    actualStdVal, ...
    actualStdTest, ...
    predictionBiasVal, ...
    predictionBiasTest, ...
    correlationVal, ...
    correlationTest, ...
    'VariableNames',{ ...
    'Model', ...
    'ValidationPredictionMean', ...
    'TestPredictionMean', ...
    'ValidationPredictionStd', ...
    'TestPredictionStd', ...
    'ValidationPredictionMin', ...
    'ValidationPredictionMax', ...
    'TestPredictionMin', ...
    'TestPredictionMax', ...
    'PredictionMeanShiftZ', ...
    'ValidationActualMean', ...
    'TestActualMean', ...
    'ValidationActualStd', ...
    'TestActualStd', ...
    'ValidationPredictionBias', ...
    'TestPredictionBias', ...
    'ValidationPredictionCorrelation', ...
    'TestPredictionCorrelation'});

writetable( ...
    predictionShiftTable, ...
    fullfile(tableDir,'STEP9_PREDICTION_SHIFT.csv'));

disp(predictionShiftTable);

%% ============================================================
% 14. SIGNAL DISTRIBUTION
% =============================================================

fprintf('\n============================================================\n');
fprintf('SIGNAL DISTRIBUTION DIAGNOSTICS\n');
fprintf('============================================================\n');

signalRows = [];

% Thresholds used in Step 8
thresholds = [ ...
    0.0074; ...
    0.0013; ...
    0.0010; ...
    0.0024; ...
    0.0063];

signalModel = strings(nModels,1);

valBUY = zeros(nModels,1);
valHOLD = zeros(nModels,1);
valSELL = zeros(nModels,1);

testBUY = zeros(nModels,1);
testHOLD = zeros(nModels,1);
testSELL = zeros(nModels,1);

for m = 1:nModels

    model = modelPrefixes{m};

    predVar = [model '_PredictedReturn'];

    if ~ismember(predVar,P.Properties.VariableNames)
        continue;
    end

    pred = double(P.(predVar));

    valMask = P.Date >= valStart & P.Date <= valEnd;
    testMask = P.Date >= testStart & P.Date <= testEnd;

    threshold = thresholds(m);

    valPred = pred(valMask);
    testPred = pred(testMask);

    valPred = valPred(isfinite(valPred));
    testPred = testPred(isfinite(testPred));

    valPosition = makePosition(valPred,threshold);
    testPosition = makePosition(testPred,threshold);

    valBUY(m) = sum(valPosition == 1);
    valHOLD(m) = sum(valPosition == 0);
    valSELL(m) = sum(valPosition == -1);

    testBUY(m) = sum(testPosition == 1);
    testHOLD(m) = sum(testPosition == 0);
    testSELL(m) = sum(testPosition == -1);

    signalModel(m) = string(model);

end

signalTable = table( ...
    signalModel, ...
    thresholds, ...
    valBUY, ...
    valHOLD, ...
    valSELL, ...
    testBUY, ...
    testHOLD, ...
    testSELL, ...
    'VariableNames',{ ...
    'Model', ...
    'Threshold', ...
    'ValidationBUY', ...
    'ValidationHOLD', ...
    'ValidationSELL', ...
    'TestBUY', ...
    'TestHOLD', ...
    'TestSELL'});

writetable( ...
    signalTable, ...
    fullfile(tableDir,'STEP9_SIGNAL_DISTRIBUTION.csv'));

disp(signalTable);

%% ============================================================
% 15. FACTOR CORRELATION MATRIX
% =============================================================

fprintf('\n============================================================\n');
fprintf('FACTOR CORRELATION / MULTICOLLINEARITY\n');
fprintf('============================================================\n');

% Use training data only for correlation diagnosis.
%
% This avoids allowing the test set to influence our understanding
% of factor redundancy.

Xtrain = X(idxTrain,:);

validColumns = false(1,nFactors);

for j = 1:nFactors

    x = Xtrain(:,j);

    validColumns(j) = ...
        sum(isfinite(x)) > 10 && ...
        std(x(isfinite(x))) > 0;

end

Xcorr = Xtrain(:,validColumns);

corrNames = factorNames(validColumns);

% Median imputation using TRAINING data only
for j = 1:size(Xcorr,2)

    x = Xcorr(:,j);

    med = median(x(isfinite(x)));

    x(~isfinite(x)) = med;

    Xcorr(:,j) = x;

end

R = corr(Xcorr,'Rows','pairwise');

corrTable = array2table( ...
    R, ...
    'VariableNames',matlab.lang.makeValidName(corrNames), ...
    'RowNames',matlab.lang.makeValidName(corrNames));

writetable( ...
    corrTable, ...
    fullfile(tableDir,'STEP9_CORRELATION_MATRIX.csv'), ...
    'WriteRowNames',true);

% Count highly correlated pairs

upperMask = triu(true(size(R)),1);

highCorr = abs(R) > 0.90 & upperMask;

numHighCorrPairs = sum(highCorr(:));

veryHighCorr = abs(R) > 0.95 & upperMask;

numVeryHighCorrPairs = sum(veryHighCorr(:));

fprintf('Factors used for correlation: %d\n',size(R,1));
fprintf('|r| > 0.90 pairs : %d\n',numHighCorrPairs);
fprintf('|r| > 0.95 pairs : %d\n',numVeryHighCorrPairs);

%% ============================================================
% 16. TOP HIGHLY CORRELATED FACTOR PAIRS
% =============================================================

fprintf('\nTop highly correlated factor pairs:\n');
fprintf('------------------------------------------------------------\n');

pairValues = [];

for i = 1:size(R,1)-1

    for j = i+1:size(R,2)

        pairValues(end+1,:) = [ ...
            i, ...
            j, ...
            R(i,j), ...
            abs(R(i,j))]; %#ok<SAGROW>

    end

end

pairValues = sortrows(pairValues,-4);

topPairs = min(15,size(pairValues,1));

for k = 1:topPairs

    i = pairValues(k,1);
    j = pairValues(k,2);

    fprintf('%2d. %-25s <-> %-25s r = %+0.4f\n', ...
        k, ...
        corrNames{i}, ...
        corrNames{j}, ...
        pairValues(k,3));

end

%% ============================================================
% 17. ACTUAL RETURN DISTRIBUTION
% =============================================================

targetName = 'TargetReturn1D';

if ismember(targetName,T.Properties.VariableNames)

    actualReturn = double(T.(targetName));

else

    error('TargetReturn1D not found.');

end

actualVal = actualReturn(idxVal);
actualTest = actualReturn(idxTest);

actualVal = actualVal(isfinite(actualVal));
actualTest = actualTest(isfinite(actualTest));

fprintf('\n============================================================\n');
fprintf('ACTUAL RETURN REGIME SHIFT\n');
fprintf('============================================================\n');

fprintf('Validation mean : %.6f\n',mean(actualVal));
fprintf('Test mean       : %.6f\n',mean(actualTest));

fprintf('Validation std  : %.6f\n',std(actualVal));
fprintf('Test std        : %.6f\n',std(actualTest));

fprintf('Validation median : %.6f\n',median(actualVal));
fprintf('Test median       : %.6f\n',median(actualTest));

fprintf('Validation positive days : %.2f%%\n', ...
    mean(actualVal>0)*100);

fprintf('Test positive days       : %.2f%%\n', ...
    mean(actualTest>0)*100);

%% ============================================================
% 18. DIAGNOSTIC SUMMARY
% =============================================================

fprintf('\n============================================================\n');
fprintf('DIAGNOSTIC SUMMARY\n');
fprintf('============================================================\n');

% Average absolute factor shift
meanAbsFactorShift = mean( ...
    abs(meanShiftTestZ), ...
    'omitnan');

% Number of large shifts
numLargeFactorShifts = sum( ...
    abs(meanShiftTestZ) > 1, ...
    'omitnan');

% Prediction shift
predictionShiftMagnitude = abs(predMeanShiftZ);

% Count models with >1 std prediction mean shift
numLargePredictionShifts = sum( ...
    predictionShiftMagnitude > 1, ...
    'omitnan');

% Actual return shift
actualMeanShiftZ = ...
    (mean(actualTest)-mean(actualVal))/std(actualVal);

actualStdRatio = ...
    std(actualTest)/std(actualVal);

% Sentiment shift
sentimentMeanShift = NaN;
sentimentStdRatio = NaN;

if ~isempty(sentimentName)

    sentimentMeanShift = ...
        (mean(testSent)-mean(valSent));

    if std(valSent) > 0
        sentimentStdRatio = ...
            std(testSent)/std(valSent);
    end

end

summary = table( ...
    meanAbsFactorShift, ...
    numLargeFactorShifts, ...
    numHighCorrPairs, ...
    numVeryHighCorrPairs, ...
    actualMeanShiftZ, ...
    actualStdRatio, ...
    sentimentMeanShift, ...
    sentimentStdRatio, ...
    numLargePredictionShifts, ...
    'VariableNames',{ ...
    'MeanAbsoluteFactorTestShiftZ', ...
    'FactorsWithAbsShiftGreaterThan1', ...
    'FactorPairsAbsCorrelationGreaterThan090', ...
    'FactorPairsAbsCorrelationGreaterThan095', ...
    'ActualReturnMeanShiftZ', ...
    'ActualReturnTestToValidationStdRatio', ...
    'SentimentMeanTestMinusValidation', ...
    'SentimentStdTestToValidationRatio', ...
    'ModelsWithPredictionMeanShiftGreaterThan1Std'});

writetable( ...
    summary, ...
    fullfile(tableDir,'STEP9_DISTRIBUTION_SHIFT_SUMMARY.csv'));

disp(summary);

%% ============================================================
% 19. TEXT DIAGNOSTIC REPORT
% =============================================================

reportFile = fullfile( ...
    tableDir, ...
    'STEP9_DIAGNOSTIC_SUMMARY.txt');

fid = fopen(reportFile,'w');

if fid == -1
    warning('Could not create diagnostic text report.');
else

    fprintf(fid,'STEP 9 - DISTRIBUTION SHIFT & PREDICTION DIAGNOSTICS\n');
    fprintf(fid,'=========================================================\n\n');

    fprintf(fid,'Validation period: %s -> %s\n', ...
        datestr(valStart,'yyyy-mm-dd'), ...
        datestr(valEnd,'yyyy-mm-dd'));

    fprintf(fid,'Test period: %s -> %s\n\n', ...
        datestr(testStart,'yyyy-mm-dd'), ...
        datestr(testEnd,'yyyy-mm-dd'));

    fprintf(fid,'FACTOR SHIFT\n');
    fprintf(fid,'------------\n');

    fprintf(fid,'Mean absolute factor test shift (Z): %.4f\n', ...
        meanAbsFactorShift);

    fprintf(fid,'Factors with absolute test shift > 1 standard deviation: %d\n', ...
        numLargeFactorShifts);

    fprintf(fid,'\nMULTICOLLINEARITY\n');
    fprintf(fid,'-----------------\n');

    fprintf(fid,'Factor pairs with |r| > 0.90: %d\n', ...
        numHighCorrPairs);

    fprintf(fid,'Factor pairs with |r| > 0.95: %d\n', ...
        numVeryHighCorrPairs);

    fprintf(fid,'\nACTUAL RETURN SHIFT\n');
    fprintf(fid,'-------------------\n');

    fprintf(fid,'Validation mean return: %.8f\n',mean(actualVal));
    fprintf(fid,'Test mean return: %.8f\n',mean(actualTest));

    fprintf(fid,'Validation std: %.8f\n',std(actualVal));
    fprintf(fid,'Test std: %.8f\n',std(actualTest));

    fprintf(fid,'Mean shift Z-score: %.4f\n',actualMeanShiftZ);
    fprintf(fid,'Test/Validation std ratio: %.4f\n',actualStdRatio);

    fprintf(fid,'\nPREDICTION SHIFT\n');
    fprintf(fid,'----------------\n');

    for m = 1:nModels

        fprintf(fid, ...
            '%s: prediction mean validation = %.8f, test = %.8f, shift Z = %.4f\n', ...
            modelPrefixes{m}, ...
            predMeanVal(m), ...
            predMeanTest(m), ...
            predMeanShiftZ(m));

        fprintf(fid, ...
            '%s: prediction bias validation = %.8f, test = %.8f\n', ...
            modelPrefixes{m}, ...
            predictionBiasVal(m), ...
            predictionBiasTest(m));

        fprintf(fid, ...
            '%s: prediction correlation validation = %.4f, test = %.4f\n', ...
            modelPrefixes{m}, ...
            correlationVal(m), ...
            correlationTest(m));

        fprintf(fid,'\n');

    end

    fprintf(fid,'\nSIGNAL SHIFT\n');
    fprintf(fid,'------------\n');

    for m = 1:nModels

        fprintf(fid, ...
            '%s: Validation BUY/HOLD/SELL = %d/%d/%d\n', ...
            modelPrefixes{m}, ...
            valBUY(m), ...
            valHOLD(m), ...
            valSELL(m));

        fprintf(fid, ...
            '%s: Test BUY/HOLD/SELL = %d/%d/%d\n', ...
            modelPrefixes{m}, ...
            testBUY(m), ...
            testHOLD(m), ...
            testSELL(m));

    end

    fclose(fid);

end

%% ============================================================
% 20. FIGURE - FACTOR SHIFT
% =============================================================

figure('Name','Step 9 Factor Distribution Shift');

plotValues = meanShiftTestZ;

[~,order] = sort(abs(plotValues),'descend');

topPlot = min(15,numel(order));

barh( ...
    plotValues(order(1:topPlot)));

yticks(1:topPlot);

yticklabels( ...
    factorNames(order(1:topPlot)));

xlabel('Test Mean Shift (Z-score relative to Train)');
ylabel('Factor');

title('Top Factor Distribution Shifts');

grid on;

saveas(gcf, ...
    fullfile(figureDir, ...
    'STEP9_FACTOR_SHIFT.png'));

%% ============================================================
% 21. FIGURE - PREDICTION DISTRIBUTIONS
% =============================================================

figure('Name','Step 9 Prediction Distributions');

hold on;

for m = 1:nModels

    model = modelPrefixes{m};

    predVar = [model '_PredictedReturn'];

    if ~ismember(predVar,P.Properties.VariableNames)
        continue;
    end

    pred = double(P.(predVar));

    valMask = P.Date >= valStart & P.Date <= valEnd;
    testMask = P.Date >= testStart & P.Date <= testEnd;

    pv = pred(valMask);
    pt = pred(testMask);

    pv = pv(isfinite(pv));
    pt = pt(isfinite(pt));

    if isempty(pv) || isempty(pt)
        continue;
    end

    % Plot mean markers rather than density to avoid
    % excessive overlap.
    plot( ...
        [mean(pv),mean(pt)], ...
        [m,m], ...
        '-o', ...
        'LineWidth',1.5);

end

yticks(1:nModels);
yticklabels(modelPrefixes);

xlabel('Prediction Mean');
ylabel('Model');

title('Validation vs Test Prediction Mean');

grid on;

saveas(gcf, ...
    fullfile(figureDir, ...
    'STEP9_PREDICTION_DISTRIBUTIONS.png'));

%% ============================================================
% 22. FIGURE - ACTUAL RETURN DISTRIBUTION
% =============================================================

figure('Name','Step 9 Actual Return Distribution');

histogram(actualVal,30, ...
    'Normalization','pdf');

hold on;

histogram(actualTest,30, ...
    'Normalization','pdf');

xlabel('1-Day Return');
ylabel('Density');

title('Validation vs Test Actual Return Distribution');

legend( ...
    {'Validation','Test'}, ...
    'Location','best');

grid on;

saveas(gcf, ...
    fullfile(figureDir, ...
    'STEP9_ACTUAL_RETURN_DISTRIBUTION.png'));

%% ============================================================
% 23. FIGURE - SENTIMENT SHIFT
% =============================================================

if ~isempty(sentimentName)

    figure('Name','Step 9 Sentiment Shift');

    histogram( ...
        trainSent, ...
        30, ...
        'Normalization','pdf');

    hold on;

    histogram( ...
        valSent, ...
        30, ...
        'Normalization','pdf');

    histogram( ...
        testSent, ...
        30, ...
        'Normalization','pdf');

    xlabel('Sentiment');
    ylabel('Density');

    title('Sentiment Distribution Across Periods');

    legend( ...
        {'Train','Validation','Test'}, ...
        'Location','best');

    grid on;

    saveas(gcf, ...
        fullfile(figureDir, ...
        'STEP9_SENTIMENT_SHIFT.png'));

end

%% ============================================================
% 24. FIGURE - FACTOR CORRELATION
% =============================================================

figure('Name','Step 9 Factor Correlation');

imagesc(R);

axis square;

colorbar;

title('Training-Period Factor Correlation Matrix');

xlabel('Factor');
ylabel('Factor');

saveas(gcf, ...
    fullfile(figureDir, ...
    'STEP9_FACTOR_CORRELATION.png'));

%% ============================================================
% 25. FIGURE - PREDICTION VS ACTUAL
% =============================================================

figure('Name','Step 9 Prediction vs Actual');

for m = 1:nModels

    model = modelPrefixes{m};

    predVar = [model '_PredictedReturn'];
    actualVar = [model '_ActualReturn'];

    if ~ismember(predVar,P.Properties.VariableNames)
        continue;
    end

    if ~ismember(actualVar,P.Properties.VariableNames)
        continue;
    end

    pred = double(P.(predVar));
    actual = double(P.(actualVar));

    testMask = ...
        P.Date >= testStart & ...
        P.Date <= testEnd;

    pred = pred(testMask);
    actual = actual(testMask);

    valid = isfinite(pred) & isfinite(actual);

    pred = pred(valid);
    actual = actual(valid);

    if isempty(pred)
        continue;
    end

    % Separate figures are preferable for research plots,
    % but this figure provides a compact diagnostic view.
    figure('Name', ...
        ['Step 9 - ' model ' Prediction vs Actual']);

    scatter(actual,pred,20,'filled');

    hold on;

    minVal = min([actual;pred]);
    maxVal = max([actual;pred]);

    plot( ...
        [minVal,maxVal], ...
        [minVal,maxVal], ...
        'LineWidth',1.2);

    xlabel('Actual 1-Day Return');
    ylabel('Predicted 1-Day Return');

    title( ...
        ['Test Prediction vs Actual - ' model]);

    grid on;

    saveas(gcf, ...
        fullfile(figureDir, ...
        ['STEP9_' model '_PREDICTION_VS_ACTUAL.png']));

end

%% ============================================================
% 26. FIGURE - SIGNAL DISTRIBUTION
% =============================================================

figure('Name','Step 9 Signal Distribution');

signalMatrix = [ ...
    testBUY, ...
    testHOLD, ...
    testSELL];

bar(signalMatrix,'stacked');

xticks(1:nModels);
xticklabels(modelPrefixes);

xtickangle(30);

ylabel('Number of Test Days');

title('Test Signal Distribution');

legend( ...
    {'BUY','HOLD','SELL'}, ...
    'Location','best');

grid on;

saveas(gcf, ...
    fullfile(figureDir, ...
    'STEP9_SIGNAL_DISTRIBUTION.png'));

%% ============================================================
% 27. FINAL CONSOLE SUMMARY
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' STEP 9 COMPLETED\n');
fprintf('============================================================\n');

fprintf('\nKey diagnostic numbers:\n');

fprintf('Mean absolute factor test shift Z : %.4f\n', ...
    meanAbsFactorShift);

fprintf('Factors with |shift| > 1           : %d / %d\n', ...
    numLargeFactorShifts,nFactors);

fprintf('Factor pairs |r| > 0.90            : %d\n', ...
    numHighCorrPairs);

fprintf('Factor pairs |r| > 0.95            : %d\n', ...
    numVeryHighCorrPairs);

fprintf('Actual-return mean shift Z         : %.4f\n', ...
    actualMeanShiftZ);

fprintf('Actual test/validation std ratio   : %.4f\n', ...
    actualStdRatio);

fprintf('\nPrediction diagnostics:\n');

for m = 1:nModels

    fprintf('%-20s mean shift Z = %+7.3f | ', ...
        modelPrefixes{m}, ...
        predMeanShiftZ(m));

    fprintf('test corr = %+7.4f | ', ...
        correlationTest(m));

    fprintf('BUY/HOLD/SELL = %d/%d/%d\n', ...
        testBUY(m), ...
        testHOLD(m), ...
        testSELL(m));

end

fprintf('\nSaved files:\n');

fprintf('%s\n', ...
    fullfile(tableDir,'STEP9_DISTRIBUTION_SHIFT_SUMMARY.csv'));

fprintf('%s\n', ...
    fullfile(tableDir,'STEP9_FACTOR_SHIFT.csv'));

fprintf('%s\n', ...
    fullfile(tableDir,'STEP9_PREDICTION_SHIFT.csv'));

fprintf('%s\n', ...
    fullfile(tableDir,'STEP9_SIGNAL_DISTRIBUTION.csv'));

fprintf('%s\n', ...
    fullfile(tableDir,'STEP9_CORRELATION_MATRIX.csv'));

fprintf('%s\n', ...
    fullfile(tableDir,'STEP9_DIAGNOSTIC_SUMMARY.txt'));

fprintf('\n============================================================\n');


%% ============================================================
% LOCAL FUNCTIONS
% ============================================================

function position = makePosition(prediction,threshold)

    prediction = prediction(:);

    position = zeros(size(prediction));

    position(prediction > threshold) = 1;

    position(prediction < -threshold) = -1;

end


function c = safeCorrelation(a,b)

    a = a(:);
    b = b(:);

    valid = isfinite(a) & isfinite(b);

    a = a(valid);
    b = b(valid);

    if numel(a) < 2
        c = NaN;
        return;
    end

    if std(a) == 0 || std(b) == 0
        c = NaN;
        return;
    end

    R = corrcoef(a,b);

    c = R(1,2);

end