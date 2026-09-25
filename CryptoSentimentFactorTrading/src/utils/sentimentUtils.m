function s = sentimentUtils()
%SENTIMENTUTILS Utility functions for sentiment classification, baseline comparison
%(VADER-style rule, Ratio-rule, and ML Classifier), and performance metrics.
%
%   S = SENTIMENTUTILS() returns a struct of function handles.

s.vaderClassify            = @vaderClassify;
s.ratioRuleClassify        = @ratioRuleClassify;
s.trainMLClassifier        = @trainMLClassifier;
s.evaluateClassifier       = @evaluateClassifier;
s.computeConfusionMatrix   = @computeConfusionMatrix;

end

%% ------------------------------------------------------------
function labels = vaderClassify(scores, posThreshold, negThreshold)
%VADERCLASSIFY Classifies sentiment into Positive, Neutral, Negative
%using compound polarity thresholding (standard VADER compound rule:
%Positive >= posThreshold (e.g. 0.05 or normalized 0.55),
%Negative <= negThreshold (e.g. -0.05 or normalized 0.45), else Neutral).

if nargin < 2 || isempty(posThreshold)
    posThreshold = 0.55; % For [0, 1] normalized scores
end
if nargin < 3 || isempty(negThreshold)
    negThreshold = 0.45;
end

labels = categorical(repmat("Neutral", size(scores)), {'Negative', 'Neutral', 'Positive'}, 'Ordinal', true);
labels(scores >= posThreshold) = 'Positive';
labels(scores <= negThreshold) = 'Negative';

end

%% ------------------------------------------------------------
function labels = ratioRuleClassify(scores, sampleSizes, window)
%RATIORULECLASSIFY Computes a volume-weighted rolling sentiment ratio baseline.
%Compares daily sentiment score against its rolling volume-weighted moving average.

if nargin < 3 || isempty(window)
    window = 14;
end

n = length(scores);
labels = categorical(repmat("Neutral", n, 1), {'Negative', 'Neutral', 'Positive'}, 'Ordinal', true);

if isempty(sampleSizes) || all(isnan(sampleSizes))
    sampleSizes = ones(n, 1);
end

% Safe sample sizes
weights = sampleSizes;
weights(isnan(weights) | weights <= 0) = 1;

weightedScore = scores .* weights;
sumWeighted = movsum(weightedScore, [window-1 0], 'omitnan');
sumWeights  = movsum(weights, [window-1 0], 'omitnan');
rollingAvg = sumWeighted ./ max(sumWeights, eps);

rollingStd = movstd(scores, [window-1 0], 'omitnan');
rollingStd(isnan(rollingStd) | rollingStd < 1e-4) = 0.05;

upperBand = rollingAvg + 0.5 * rollingStd;
lowerBand = rollingAvg - 0.5 * rollingStd;

labels(scores > upperBand) = 'Positive';
labels(scores < lowerBand) = 'Negative';

end

%% ------------------------------------------------------------
function model = trainMLClassifier(XTrain, yTrainLabels)
%TRAINMLCLASSIFIER Trains a Decision Tree / Ensemble classification model
%for sentiment-driven directional prediction.

try
    model = fitctree(XTrain, yTrainLabels, 'MaxNumSplits', 20, 'CrossVal', 'off');
catch
    % Fallback if Statistics toolbox function fitctree is called differently
    model = fitcsvm(XTrain, yTrainLabels);
end

end

%% ------------------------------------------------------------
function metrics = evaluateClassifier(trueLabels, predLabels)
%EVALUATECLASSIFIER Evaluates 3-class or 2-class predictions: Accuracy,
%Precision, Recall, F1-Score per class and macro-average.

trueCat = categorical(trueLabels);
predCat = categorical(predLabels);
categoriesList = categories(trueCat);

confMat = confusionmat(trueCat, predCat);

total = sum(confMat(:));
accuracy = sum(diag(confMat)) / max(total, 1);

nClasses = numel(categoriesList);
prec = zeros(nClasses, 1);
rec = zeros(nClasses, 1);
f1 = zeros(nClasses, 1);

for c = 1:nClasses
    tp = confMat(c, c);
    fp = sum(confMat(:, c)) - tp;
    fn = sum(confMat(c, :)) - tp;
    
    if (tp + fp) > 0
        prec(c) = tp / (tp + fp);
    else
        prec(c) = 0;
    end
    
    if (tp + fn) > 0
        rec(c) = tp / (tp + fn);
    else
        rec(c) = 0;
    end
    
    if (prec(c) + rec(c)) > 0
        f1(c) = 2 * (prec(c) * rec(c)) / (prec(c) + rec(c));
    else
        f1(c) = 0;
    end
end

metrics = struct();
metrics.confusionMatrix = confMat;
metrics.classes         = categoriesList;
metrics.accuracy        = accuracy;
metrics.precision       = prec;
metrics.recall          = rec;
metrics.f1Score         = f1;
metrics.macroF1         = mean(f1);

end

%% ------------------------------------------------------------
function cm = computeConfusionMatrix(trueLabels, predLabels)
%COMPUTECONFUSIONMATRIX Computes a standard confusion matrix.
cm = confusionmat(categorical(trueLabels), categorical(predLabels));
end
