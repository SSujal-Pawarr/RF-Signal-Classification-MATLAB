function net = trainRFModel()

clc;

%% ============================================================
% PROJECT PATHS
% ============================================================

projectRoot = fileparts(fileparts(mfilename('fullpath')));

datasetRoot = fullfile( ...
    projectRoot, ...
    "data", ...
    "dataset");

%% ============================================================
% 8 RF SIGNAL CLASSES
% ============================================================

classes = [
    "fm"
    "bluetooth"
    "wifi"
    "cellular"
    "lora"
    "ais"
    "airband"
    "RS41-Radiosonde"
];

%% ============================================================
% LOAD TRAINING DATASET
% ============================================================

imdsTrain = imageDatastore( ...
    fullfile(datasetRoot, "train"), ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

%% ============================================================
% LOAD VALIDATION DATASET
% ============================================================

imdsValidation = imageDatastore( ...
    fullfile(datasetRoot, "validation"), ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

%% ============================================================
% KEEP ONLY SELECTED 8 CLASSES
% ============================================================

imdsTrain = subset( ...
    imdsTrain, ...
    ismember(string(imdsTrain.Labels), classes));

imdsValidation = subset( ...
    imdsValidation, ...
    ismember(string(imdsValidation.Labels), classes));

%% ============================================================
% DISPLAY DATASET INFORMATION
% ============================================================

disp("Training images:");
disp(countEachLabel(imdsTrain));

disp("Validation images:");
disp(countEachLabel(imdsValidation));

%% ============================================================
% LOAD PRETRAINED MOBILENETV2
% ============================================================

net = imagePretrainedNetwork("mobilenetv2");

inputSize = net.Layers(1).InputSize;

disp("Network input size:");
disp(inputSize);

%% ============================================================
% CONVERT NETWORK TO LAYER GRAPH
% ============================================================

lgraph = layerGraph(net);

%% ============================================================
% FIND FINAL FULLY CONNECTED LAYER
% ============================================================

layers = lgraph.Layers;

fcIndex = [];

for i = 1:numel(layers)

    if isa( ...
            layers(i), ...
            "nnet.cnn.layer.FullyConnectedLayer")

        fcIndex = i;

    end

end

if isempty(fcIndex)

    error( ...
        "Could not find the final fully connected layer.");

end

oldFCName = layers(fcIndex).Name;

fprintf( ...
    "Replacing classifier: %s\n", ...
    oldFCName);

%% ============================================================
% REPLACE FINAL FULLY CONNECTED LAYER
% ============================================================

newFC = fullyConnectedLayer( ...
    numel(classes), ...
    "Name", "rf_classifier", ...
    "WeightLearnRateFactor", 10, ...
    "BiasLearnRateFactor", 10);

lgraph = replaceLayer( ...
    lgraph, ...
    oldFCName, ...
    newFC);

%% ============================================================
% REPLACE ORIGINAL SOFTMAX LAYER
% ============================================================

newSoftmax = softmaxLayer( ...
    "Name", "rf_softmax");

lgraph = replaceLayer( ...
    lgraph, ...
    "Logits_softmax", ...
    newSoftmax);

%% ============================================================
% ADD CLASSIFICATION OUTPUT LAYER
% ============================================================

outputLayer = classificationLayer( ...
    "Name", "rf_output", ...
    "Classes", categorical(classes));

lgraph = addLayers( ...
    lgraph, ...
    outputLayer);

%% ============================================================
% CONNECT SOFTMAX TO CLASSIFICATION OUTPUT
% ============================================================

lgraph = connectLayers( ...
    lgraph, ...
    "rf_softmax", ...
    "rf_output");

%% ============================================================
% TRAINING IMAGE PREPARATION
% ============================================================

%
% Training images receive:
%
% 1. AWGN noise
% 2. Brightness jitter
% 3. Contrast jitter
%
% Output:
%
% 224 x 224 x 3
% single precision
% values in [0,1]
%

imdsTrain.ReadFcn = @augmentRFSpectrogram;

%% ============================================================
% VALIDATION IMAGE PREPARATION
% ============================================================

%
% Validation images receive NO random augmentation.
%
% They are converted to:
%
% RGB
% 224 x 224
% single precision
% [0,1]
%

imdsValidation.ReadFcn = @prepareRFValidationImage;

%% ============================================================
% EXISTING GEOMETRIC AUGMENTATION
% ============================================================

imageAugmenter = imageDataAugmenter( ...
    "RandXTranslation", [-7 7], ...
    "RandYTranslation", [-7 7], ...
    "RandXReflection", false);

%% ============================================================
% AUGMENTED TRAINING DATASTORE
% ============================================================

augmentedTrain = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsTrain, ...
    "DataAugmentation", imageAugmenter);

%% ============================================================
% VALIDATION DATASTORE
% ============================================================

augmentedValidation = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsValidation);

%% ============================================================
% TRAINING OPTIONS
% ============================================================

options = trainingOptions("adam", ...
    "InitialLearnRate", 1e-4, ...
    "MaxEpochs", 10, ...
    "MiniBatchSize", 32, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", augmentedValidation, ...
    "ValidationFrequency", 20, ...
    "Verbose", true, ...
    "Plots", "none");

%% ============================================================
% START TRAINING
% ============================================================

disp(" ");
disp("====================================");
disp("STARTING RF SIGNAL TRAINING");
disp("====================================");

disp("RF-specific augmentation enabled:");
disp(" - AWGN noise");
disp(" - Random brightness jitter");
disp(" - Random contrast jitter");
disp(" - Existing X/Y translation");

disp(" ");
disp("Validation preprocessing:");
disp(" - RGB conversion");
disp(" - Resize to 224 x 224");
disp(" - Single precision");
disp(" - Pixel range [0,1]");
disp(" - No random augmentation");

disp(" ");
disp("Training on single CPU.");

%% ============================================================
% TRAIN NETWORK
% ============================================================

net = trainNetwork( ...
    augmentedTrain, ...
    lgraph, ...
    options);

%% ============================================================
% CREATE MODEL FOLDER
% ============================================================

modelFolder = fullfile( ...
    projectRoot, ...
    "models");

if ~exist(modelFolder, "dir")

    mkdir(modelFolder);

end

%% ============================================================
% SAVE TRAINED MODEL
% ============================================================

modelPath = fullfile( ...
    modelFolder, ...
    "rfSignalNet.mat");

save( ...
    modelPath, ...
    "net");

disp(" ");
disp("New trained model saved:");
disp(modelPath);

%% ============================================================
% VALIDATION EVALUATION
% ============================================================

disp(" ");
disp("====================================");
disp("STARTING VALIDATION EVALUATION");
disp("====================================");

reset(augmentedValidation);

%% ============================================================
% CLASSIFY VALIDATION IMAGES
% ============================================================

[YPredRaw, validationScores] = ...
    classify( ...
        net, ...
        augmentedValidation);

%% ============================================================
% GET TRUE LABELS
% ============================================================

YTrueRaw = imdsValidation.Labels;

%% ============================================================
% CONVERT LABELS
% ============================================================

YTrue = categorical( ...
    string(YTrueRaw), ...
    classes);

YPred = categorical( ...
    string(YPredRaw), ...
    classes);

%% ============================================================
% VERIFY VALIDATION ALIGNMENT
% ============================================================

fprintf("\n");

fprintf( ...
    "Validation labels: %d\n", ...
    numel(YTrue));

fprintf( ...
    "Predictions       : %d\n", ...
    numel(YPred));

fprintf( ...
    "Score rows        : %d\n", ...
    size(validationScores, 1));

if numel(YTrue) ~= numel(YPred)

    error( ...
        "Ground-truth labels and predictions have different lengths.");

end

if size(validationScores, 1) ~= numel(YTrue)

    error( ...
        "Number of score rows does not match validation labels.");

end

%% ============================================================
% CONFUSION MATRIX
% ============================================================

classCategories = categorical(classes);

confusionMatrix = confusionmat( ...
    YTrue, ...
    YPred, ...
    "Order", ...
    classCategories);

disp(" ");
disp("Confusion Matrix:");
disp(confusionMatrix);

%% ============================================================
% PER-CLASS METRICS
% ============================================================

numClasses = numel(classes);

precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
f1Score = zeros(numClasses, 1);
support = zeros(numClasses, 1);

for i = 1:numClasses

    %% True Positive

    truePositive = ...
        confusionMatrix(i, i);

    %% False Positive

    falsePositive = ...
        sum(confusionMatrix(:, i)) - ...
        truePositive;

    %% False Negative

    falseNegative = ...
        sum(confusionMatrix(i, :)) - ...
        truePositive;

    %% Support

    support(i) = ...
        sum(confusionMatrix(i, :));

    %% Precision

    if ...
            (truePositive + falsePositive) > 0

        precision(i) = ...
            truePositive / ...
            (truePositive + falsePositive);

    else

        precision(i) = 0;

    end

    %% Recall

    if ...
            (truePositive + falseNegative) > 0

        recall(i) = ...
            truePositive / ...
            (truePositive + falseNegative);

    else

        recall(i) = 0;

    end

    %% F1 Score

    if ...
            (precision(i) + recall(i)) > 0

        f1Score(i) = ...
            2 * precision(i) * recall(i) / ...
            (precision(i) + recall(i));

    else

        f1Score(i) = 0;

    end

end

%% ============================================================
% FORCE ALL METRICS TO COLUMN VECTORS
% ============================================================

classesColumn = string(classes(:));

support = double(support(:));

precision = double(precision(:));

recall = double(recall(:));

f1Score = double(f1Score(:));

%% ============================================================
% CONVERT TO PERCENTAGE
% ============================================================

precisionPercent = ...
    precision * 100;

recallPercent = ...
    recall * 100;

f1Percent = ...
    f1Score * 100;

%% ============================================================
% OVERALL ACCURACY
% ============================================================

correctPredictions = ...
    sum(diag(confusionMatrix));

totalPredictions = ...
    sum(confusionMatrix(:));

incorrectPredictions = ...
    totalPredictions - ...
    correctPredictions;

overallAccuracy = ...
    correctPredictions / ...
    totalPredictions;

overallAccuracyPercent = ...
    overallAccuracy * 100;

%% ============================================================
% DISPLAY VALIDATION RESULTS
% ============================================================

fprintf("\n");
fprintf("====================================\n");
fprintf("VALIDATION RESULTS\n");
fprintf("====================================\n");

fprintf( ...
    "Total validation images : %d\n", ...
    totalPredictions);

fprintf( ...
    "Correct predictions     : %d\n", ...
    correctPredictions);

fprintf( ...
    "Incorrect predictions   : %d\n", ...
    incorrectPredictions);

fprintf( ...
    "Overall accuracy        : %.2f%%\n", ...
    overallAccuracyPercent);

%% ============================================================
% CREATE METRICS TABLE
% ============================================================

%
% IMPORTANT FIX:
% Every variable is explicitly converted to an 8 x 1 vector.
%

metricsTable = table( ...
    classesColumn, ...
    support, ...
    precisionPercent, ...
    recallPercent, ...
    f1Percent, ...
    'VariableNames', { ...
        'Class', ...
        'Support', ...
        'Precision', ...
        'Recall', ...
        'F1_Score'});

disp(" ");
disp("Per-Class Validation Metrics:");
disp(metricsTable);

%% ============================================================
% MACRO METRICS
% ============================================================

macroPrecision = ...
    mean(precisionPercent);

macroRecall = ...
    mean(recallPercent);

macroF1 = ...
    mean(f1Percent);

fprintf("\n");

fprintf( ...
    "Macro Precision : %.2f%%\n", ...
    macroPrecision);

fprintf( ...
    "Macro Recall    : %.2f%%\n", ...
    macroRecall);

fprintf( ...
    "Macro F1-Score  : %.2f%%\n", ...
    macroF1);

%% ============================================================
% CREATE RESULTS FOLDER
% ============================================================

resultsFolder = fullfile( ...
    projectRoot, ...
    "results");

if ~exist(resultsFolder, "dir")

    mkdir(resultsFolder);

end

%% ============================================================
% PER-CLASS UNKNOWN THRESHOLD CALIBRATION
% ============================================================

disp(" ");
disp("====================================");
disp("CALIBRATING PER-CLASS THRESHOLDS");
disp("====================================");

%% Threshold percentile

thresholdPercentile = 10;

%% Storage

classThresholds = ...
    zeros(numClasses, 1);

correctSampleCount = ...
    zeros(numClasses, 1);

minimumCorrectConfidence = ...
    zeros(numClasses, 1);

maximumCorrectConfidence = ...
    zeros(numClasses, 1);

%% ============================================================
% CALCULATE THRESHOLD FOR EVERY CLASS
% ============================================================

for i = 1:numClasses

    %% True class samples

    trueClassMask = ...
        string(YTrue) == classes(i);

    %% Predicted class samples

    predictedClassMask = ...
        string(YPred) == classes(i);

    %% Correct samples

    correctMask = ...
        trueClassMask & ...
        predictedClassMask;

    %% Correct-class confidence

    correctConfidences = ...
        validationScores( ...
            correctMask, ...
            i);

    %% Number of correct examples

    correctSampleCount(i) = ...
        numel(correctConfidences);

    if ~isempty(correctConfidences)

        %% Threshold

        classThresholds(i) = ...
            prctile( ...
                correctConfidences, ...
                thresholdPercentile);

        %% Minimum confidence

        minimumCorrectConfidence(i) = ...
            min(correctConfidences);

        %% Maximum confidence

        maximumCorrectConfidence(i) = ...
            max(correctConfidences);

    else

        %% Fallback

        classThresholds(i) = ...
            0.70;

        minimumCorrectConfidence(i) = ...
            0;

        maximumCorrectConfidence(i) = ...
            0;

    end

    fprintf( ...
        "%s : %.2f%% threshold | %d correct samples\n", ...
        classes(i), ...
        classThresholds(i) * 100, ...
        correctSampleCount(i));

end

%% ============================================================
% FORCE THRESHOLD VECTORS TO COLUMN FORMAT
% ============================================================

correctSampleCount = ...
    double(correctSampleCount(:));

minimumCorrectConfidence = ...
    double(minimumCorrectConfidence(:));

maximumCorrectConfidence = ...
    double(maximumCorrectConfidence(:));

classThresholds = ...
    double(classThresholds(:));

%% ============================================================
% THRESHOLD TABLE
% ============================================================

thresholdTable = table( ...
    classesColumn, ...
    correctSampleCount, ...
    minimumCorrectConfidence * 100, ...
    maximumCorrectConfidence * 100, ...
    classThresholds * 100, ...
    'VariableNames', { ...
        'Class', ...
        'CorrectSamples', ...
        'MinimumCorrectConfidence', ...
        'MaximumCorrectConfidence', ...
        'ThresholdPercent'});

disp(" ");
disp("Per-Class Confidence Thresholds:");
disp(thresholdTable);

%% ============================================================
% SAVE THRESHOLDS MAT
% ============================================================

thresholdMATPath = fullfile( ...
    resultsFolder, ...
    "per_class_thresholds.mat");

thresholdClasses = classesColumn;

save( ...
    thresholdMATPath, ...
    "classThresholds", ...
    "thresholdClasses", ...
    "thresholdPercentile", ...
    "correctSampleCount", ...
    "minimumCorrectConfidence", ...
    "maximumCorrectConfidence");

%% ============================================================
% SAVE THRESHOLDS CSV
% ============================================================

thresholdCSVPath = fullfile( ...
    resultsFolder, ...
    "per_class_thresholds.csv");

writetable( ...
    thresholdTable, ...
    thresholdCSVPath);

%% ============================================================
% SAVE METRICS CSV
% ============================================================

metricsCSVPath = fullfile( ...
    resultsFolder, ...
    "validation_metrics.csv");

writetable( ...
    metricsTable, ...
    metricsCSVPath);

%% ============================================================
% SAVE SUMMARY CSV
% ============================================================

summaryTable = table( ...
    double(overallAccuracyPercent), ...
    double(macroPrecision), ...
    double(macroRecall), ...
    double(macroF1), ...
    double(thresholdPercentile), ...
    'VariableNames', { ...
        'OverallAccuracy', ...
        'MacroPrecision', ...
        'MacroRecall', ...
        'MacroF1', ...
        'ThresholdPercentile'});

summaryCSVPath = fullfile( ...
    resultsFolder, ...
    "validation_summary.csv");

writetable( ...
    summaryTable, ...
    summaryCSVPath);

%% ============================================================
% SAVE CONFUSION MATRIX CSV
% ============================================================

confusionCSVPath = fullfile( ...
    resultsFolder, ...
    "confusion_matrix.csv");

confusionTable = array2table( ...
    confusionMatrix, ...
    'VariableNames', ...
    matlab.lang.makeValidName( ...
        cellstr(classes)), ...
    'RowNames', ...
    cellstr(classes));

writetable( ...
    confusionTable, ...
    confusionCSVPath, ...
    'WriteRowNames', true);

%% ============================================================
% SAVE CONFUSION MATRIX MAT
% ============================================================

confusionMATPath = fullfile( ...
    resultsFolder, ...
    "confusion_matrix.mat");

save( ...
    confusionMATPath, ...
    "confusionMatrix", ...
    "classes", ...
    "YTrue", ...
    "YPred");

%% ============================================================
% SAVE VALIDATION SUMMARY MAT
% ============================================================

summaryMATPath = fullfile( ...
    resultsFolder, ...
    "validation_summary.mat");

save( ...
    summaryMATPath, ...
    "overallAccuracy", ...
    "overallAccuracyPercent", ...
    "correctPredictions", ...
    "incorrectPredictions", ...
    "totalPredictions", ...
    "macroPrecision", ...
    "macroRecall", ...
    "macroF1");

%% ============================================================
% CREATE CONFUSION MATRIX FIGURE
% ============================================================

disp(" ");
disp("Creating confusion matrix figure...");

figCM = figure( ...
    "Name", ...
    "RF Signal Classification Confusion Matrix", ...
    "NumberTitle", ...
    "off");

%% IMPORTANT:
% Do NOT use:
%
% confusionchart(..., "Order", ...)
%
% because your MATLAB version reported that this
% property is unsupported.

cmChart = confusionchart( ...
    confusionMatrix, ...
    cellstr(classes));

title( ...
    "RF Signal Classification - Validation Confusion Matrix");

%% ============================================================
% SAVE CONFUSION MATRIX FIGURE
% ============================================================

confusionFigurePath = fullfile( ...
    resultsFolder, ...
    "confusion_matrix.png");

exportgraphics( ...
    figCM, ...
    confusionFigurePath, ...
    "Resolution", ...
    300);

close(figCM);

%% ============================================================
% FINAL DISPLAY
% ============================================================

disp(" ");
disp("====================================");
disp("VALIDATION RESULTS SAVED");
disp("====================================");

disp("Metrics CSV:");
disp(metricsCSVPath);

disp("Confusion matrix CSV:");
disp(confusionCSVPath);

disp("Confusion matrix MAT:");
disp(confusionMATPath);

disp("Confusion matrix PNG:");
disp(confusionFigurePath);

disp("Validation summary CSV:");
disp(summaryCSVPath);

disp("Validation summary MAT:");
disp(summaryMATPath);

disp("Per-class thresholds MAT:");
disp(thresholdMATPath);

disp("Per-class thresholds CSV:");
disp(thresholdCSVPath);

disp(" ");
disp("====================================");
disp("TRAINING COMPLETE");
disp("====================================");

disp("Model saved to:");
disp(modelPath);

disp(" ");
disp("Per-class UNKNOWN detection thresholds");
disp("have been calibrated from the validation set.");

end


%% ============================================================
% LOCAL FUNCTION:
% DETERMINISTIC VALIDATION IMAGE PREPROCESSING
% ============================================================

function I = prepareRFValidationImage(filename)

%% Read image

I = imread(filename);

%% ============================================================
% CONVERT TO RGB
% ============================================================

if ndims(I) == 2

    %% Grayscale -> RGB

    I = cat( ...
        3, ...
        I, ...
        I, ...
        I);

elseif size(I, 3) == 4

    %% RGBA -> RGB

    I = I(:, :, 1:3);

elseif size(I, 3) ~= 3

    error( ...
        "RF validation image must have 1, 3, or 4 channels.");

end

%% ============================================================
% CONVERT TO SINGLE [0,1]
% ============================================================

I = im2single(I);

%% ============================================================
% RESIZE
% ============================================================

I = imresize( ...
    I, ...
    [224 224]);

%% ============================================================
% VERIFY
% ============================================================

if ~isequal(size(I), [224 224 3])

    error( ...
        "Validation preprocessing failed. Expected 224 x 224 x 3.");

end

end


%% ============================================================
% LOCAL FUNCTION:
% RF SPECTROGRAM TRAINING AUGMENTATION
% ============================================================

function I = augmentRFSpectrogram(filename)

%% Read image

I = imread(filename);

%% ============================================================
% CONVERT TO RGB
% ============================================================

if ndims(I) == 2

    %% Grayscale -> RGB

    I = cat( ...
        3, ...
        I, ...
        I, ...
        I);

elseif size(I, 3) == 4

    %% RGBA -> RGB

    I = I(:, :, 1:3);

elseif size(I, 3) ~= 3

    error( ...
        "RF image must have 1, 3, or 4 channels.");

end

%% ============================================================
% CONVERT TO SINGLE [0,1]
% ============================================================

I = im2single(I);

%% ============================================================
% RESIZE
% ============================================================

I = imresize( ...
    I, ...
    [224 224]);

%% ============================================================
% 1. AWGN NOISE
% ============================================================

%
% Random SNR between 5 dB and 30 dB.
%

snrDB = ...
    5 + ...
    (30 - 5) * rand();

%% Signal power

signalPower = ...
    mean(I(:).^2);

%% Avoid zero signal power

if signalPower < 1e-8

    signalPower = 1e-8;

end

%% Noise power

noisePower = ...
    signalPower / ...
    (10^(snrDB / 10));

%% Generate Gaussian noise

noise = ...
    sqrt(noisePower) .* ...
    randn(size(I), "like", I);

%% Add noise

I = ...
    I + ...
    noise;

%% Clip

I = ...
    min(max(I, 0), 1);

%% ============================================================
% 2. RANDOM BRIGHTNESS JITTER
% ============================================================

brightnessShift = ...
    -0.15 + ...
    (0.30 * rand());

I = ...
    I + ...
    brightnessShift;

%% Clip

I = ...
    min(max(I, 0), 1);

%% ============================================================
% 3. RANDOM CONTRAST JITTER
% ============================================================

contrastFactor = ...
    0.80 + ...
    (1.20 - 0.80) * rand();

%% Image mean

imageMean = ...
    mean(I(:));

%% Apply contrast

I = ...
    (I - imageMean) .* ...
    contrastFactor + ...
    imageMean;

%% Clip

I = ...
    min(max(I, 0), 1);

%% ============================================================
% FINAL SIZE CHECK
% ============================================================

if ~isequal(size(I), [224 224 3])

    error( ...
        "Training augmentation failed. Expected 224 x 224 x 3.");

end

end