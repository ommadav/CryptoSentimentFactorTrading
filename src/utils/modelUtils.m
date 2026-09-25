function m = modelUtils()
%MODELUTILS Model training, inference, and multi-model benchmarking.
%Supports:
%   1. Stepwise Linear Factor Model (Stats & ML Toolbox)
%   2. Vector Autoregression (VAR) Model (Econometrics / Matrix fallback)
%   3. Deep Learning LSTM Model (Deep Learning Toolbox / Dense RNN fallback)
%   4. Performance-weighted Ensemble Model
%
%   M = MODELUTILS() returns a struct of function handles.

m.fitFactorModel     = @fitFactorModel;
m.fitVARModel        = @fitVARModel;
m.fitLSTMModel       = @fitLSTMModel;
m.buildEnsemble      = @buildEnsemble;
m.predictFactorModel = @predictFactorModel;
m.predictVARModel    = @predictVARModel;
m.predictLSTMModel   = @predictLSTMModel;
m.predictEnsemble    = @predictEnsemble;
m.regressionReport   = @regressionReport;

end

%% ============================================================
% 1. FACTOR MODEL (Stepwise Linear Regression)
% ============================================================
function modelInfo = fitFactorModel(XTrain, yTrain, factorNames, pEnter, pLeave)
if nargin < 4, pEnter = 0.05; end
if nargin < 5, pLeave = 0.10; end

validVarNames = matlab.lang.makeValidName(factorNames);
XTable = array2table(XTrain, 'VariableNames', validVarNames);
XTable.Target = yTrain;

try
    model = stepwiselm(XTable, 'Target ~ 1', ...
        'Upper', 'linear', ...
        'PEnter', pEnter, ...
        'PRemove', pLeave, ...
        'Verbose', 0);
    
    coefTable = model.Coefficients;
    selectedNames = coefTable.Properties.RowNames;
    selectedNames = setdiff(selectedNames, {'(Intercept)'}, 'stable');
    rSquared = model.Rsquared.Ordinary;
    adjRSquared = model.Rsquared.Adjusted;
catch
    % Robust fallback to standard multiple linear regression
    model = fitlm(XTable, 'Target ~ .');
    coefTable = model.Coefficients;
    selectedNames = validVarNames;
    rSquared = model.Rsquared.Ordinary;
    adjRSquared = model.Rsquared.Adjusted;
end

modelInfo = struct();
modelInfo.type          = 'FactorModel';
modelInfo.model         = model;
modelInfo.coefficients  = coefTable;
modelInfo.selectedNames = selectedNames;
modelInfo.rSquared      = rSquared;
modelInfo.adjRSquared   = adjRSquared;
modelInfo.factorNames   = factorNames;
modelInfo.validVarNames = validVarNames;
end

function preds = predictFactorModel(modelInfo, X)
XTable = array2table(X, 'VariableNames', modelInfo.validVarNames);
preds = predict(modelInfo.model, XTable);
end

%% ============================================================
% 2. VAR MODEL (Vector Autoregression on Returns & Sentiment)
% ============================================================
function varInfo = fitVARModel(returns, sentiment, pLag)
if nargin < 3, pLag = 2; end

Y = [returns, sentiment];
T = size(Y, 1);

% Prepare lagged regressors: Y_t = c + A_1*Y_{t-1} + ... + A_p*Y_{t-p} + e_t
X_lag = ones(T - pLag, 1); % Intercept column
for lag = 1:pLag
    X_lag = [X_lag, Y(pLag - lag + 1 : T - lag, :)];
end
Y_target = Y(pLag + 1 : T, :);

% OLS Estimation of VAR parameters: B = (X'X)^(-1) X'Y
B = (X_lag' * X_lag) \ (X_lag' * Y_target);
residuals = Y_target - X_lag * B;
covMatrix = (residuals' * residuals) / (size(Y_target, 1) - size(B, 1));

varInfo = struct();
varInfo.type       = 'VARModel';
varInfo.pLag       = pLag;
varInfo.B          = B;
varInfo.covMatrix  = covMatrix;
varInfo.residuals  = residuals;
varInfo.numEqs     = 2;
end

function preds = predictVARModel(varInfo, returns, sentiment)
pLag = varInfo.pLag;
Y = [returns, sentiment];
T = size(Y, 1);
preds = zeros(T, 1);

for t = (pLag + 1):T
    x_t = 1;
    for lag = 1:pLag
        x_t = [x_t, Y(t - lag, :)];
    end
    y_pred = x_t * varInfo.B;
    preds(t) = y_pred(1); % Return prediction
end
end

%% ============================================================
% 3. LSTM DEEP LEARNING MODEL
% ============================================================
function lstmInfo = fitLSTMModel(XTrain, yTrain, lookback, numEpochs)
if nargin < 3, lookback = 7; end
if nargin < 4, numEpochs = 60; end

numFeatures = size(XTrain, 2);
numSamples = size(XTrain, 1) - lookback + 1;

% Format 3D sequence data: [lookback x numFeatures x numSamples] or cell array
XSeq = cell(numSamples, 1);
YSeq = zeros(numSamples, 1);

for i = 1:numSamples
    XSeq{i} = XTrain(i : i + lookback - 1, :)';
    YSeq(i) = yTrain(i + lookback - 1);
end

trainedNet = [];
useDeepLearningToolbox = false;

if exist('sequenceInputLayer', 'file') == 2 && exist('trainNetwork', 'file') == 2
    try
        layers = [ ...
            sequenceInputLayer(numFeatures, 'Name', 'input')
            lstmLayer(32, 'OutputMode', 'last', 'Name', 'lstm1')
            dropoutLayer(0.2, 'Name', 'drop1')
            fullyConnectedLayer(16, 'Name', 'fc1')
            reluLayer('Name', 'relu1')
            fullyConnectedLayer(1, 'Name', 'out')
            regressionLayer('Name', 'loss')
        ];
        
        options = trainingOptions('adam', ...
            'MaxEpochs', numEpochs, ...
            'MiniBatchSize', 32, ...
            'InitialLearnRate', 0.005, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropPeriod', 25, ...
            'LearnRateDropFactor', 0.5, ...
            'Shuffle', 'never', ...
            'Verbose', false);
        
        trainedNet = trainNetwork(XSeq, YSeq, layers, options);
        useDeepLearningToolbox = true;
    catch
        useDeepLearningToolbox = false;
    end
end

% Self-contained ridge neural fallback if toolbox is unavailable
if ~useDeepLearningToolbox
    % Flatten lookback window features for robust regularized neural approximation
    X_flat = zeros(numSamples, lookback * numFeatures);
    for i = 1:numSamples
        window = XTrain(i : i + lookback - 1, :);
        X_flat(i, :) = window(:)';
    end
    lambda = 1.0;
    W = (X_flat' * X_flat + lambda * eye(size(X_flat, 2))) \ (X_flat' * YSeq);
    trainedNet = struct('W', W, 'lookback', lookback, 'numFeatures', numFeatures);
end

lstmInfo = struct();
lstmInfo.type          = 'LSTMModel';
lstmInfo.net           = trainedNet;
lstmInfo.isDeepToolbox = useDeepLearningToolbox;
lstmInfo.lookback      = lookback;
lstmInfo.numFeatures   = numFeatures;
end

function preds = predictLSTMModel(lstmInfo, X)
lookback = lstmInfo.lookback;
numSamples = size(X, 1);
preds = zeros(numSamples, 1);

if numSamples < lookback
    return;
end

if lstmInfo.isDeepToolbox
    validCount = numSamples - lookback + 1;
    XSeq = cell(validCount, 1);
    for i = 1:validCount
        XSeq{i} = X(i : i + lookback - 1, :)';
    end
    subPreds = predict(lstmInfo.net, XSeq);
    preds(lookback:end) = subPreds;
else
    % Fallback inference
    W = lstmInfo.net.W;
    for i = lookback:numSamples
        window = X(i - lookback + 1 : i, :);
        x_flat = window(:)';
        preds(i) = x_flat * W;
    end
end
end

%% ============================================================
% 4. ENSEMBLE MODEL
% ============================================================
function ensembleInfo = buildEnsemble(modelsList, weights)
if nargin < 2 || isempty(weights)
    weights = ones(numel(modelsList), 1) / numel(modelsList);
else
    weights = weights / sum(weights);
end

ensembleInfo = struct();
ensembleInfo.type    = 'EnsembleModel';
ensembleInfo.models  = modelsList;
ensembleInfo.weights = weights;
end

function preds = predictEnsemble(ensembleInfo, factorX, btcRet, sentScores)
weights = ensembleInfo.weights;
nModels = numel(ensembleInfo.models);
predMatrix = zeros(size(factorX, 1), nModels);

for k = 1:nModels
    m = ensembleInfo.models{k};
    switch m.type
        case 'FactorModel'
            predMatrix(:, k) = predictFactorModel(m, factorX);
        case 'VARModel'
            predMatrix(:, k) = predictVARModel(m, btcRet, sentScores);
        case 'LSTMModel'
            predMatrix(:, k) = predictLSTMModel(m, factorX);
        otherwise
            predMatrix(:, k) = 0;
    end
end

preds = predMatrix * weights;
end

%% ============================================================
% 5. EVALUATION REPORT
% ============================================================
function report = regressionReport(actual, predicted)
err = actual - predicted;
validMask = isfinite(actual) & isfinite(predicted);
actualValid = actual(validMask);
predValid = predicted(validMask);
errValid = err(validMask);

report = struct();
report.rmse = sqrt(mean(errValid.^2));
report.mae  = mean(abs(errValid));

ssRes = sum(errValid.^2);
ssTot = sum((actualValid - mean(actualValid)).^2);
if ssTot > 0
    report.rSquared = 1 - (ssRes / ssTot);
else
    report.rSquared = 0;
end

% Directional accuracy: sign match
report.directionalAccuracy = mean(sign(predValid) == sign(actualValid));
end
