function evaluateRFModel()

clc;

%% ============================================================
% PROJECT PATHS
% ============================================================

projectRoot = fileparts(fileparts(mfilename('fullpath')));

modelPath = fullfile( ...
    projectRoot, ...
    "models", ...
    "rfSignalNet.mat");

datasetRoot = fullfile( ...
    projectRoot, ...
    "data", ...
    "dataset");

validationPath = fullfile( ...
    datasetRoot, ...
    "validation");

resultsFolder = fullfile( ...
    projectRoot, ...
    "results");

if ~exist(resultsFolder, "dir")
    mkdir(resultsFolder);
end

%% ============================================================
% LOAD MODEL
% ============================================================

fprintf("\n");
fprintf("====================================\n");
fprintf("RF MODEL EVALUATION\n");
fprintf("====================================\n\n");

if ~isfile(modelPath)

    error( ...
        "Model file not found: %s", ...
        modelPath);

end

data = load( ...
    modelPath, ...
    "net");

net = data.net;

disp("Trained model loaded successfully.");

%% ============================================================
% LOAD VALIDATION DATASET
% ============================================================

imdsValidation = imageDatastore( ...
    validationPath, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

%% ============================================================
% RF CLASSES
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
% KEEP ONLY SELECTED CLASSES
% ============================================================

imdsValidation = subset( ...
    imdsValidation, ...
    ismember( ...
        string(imdsValidation.Labels), ...
        classes));

%% ============================================================
% DISPLAY VALIDATION DATASET
% ============================================================

disp(" ");
disp("Validation images:");

disp( ...
    countEachLabel(imdsValidation));

%% ============================================================
% MODEL INPUT SIZE
% ============================================================

inputSize = net.Layers(1).InputSize;

fprintf("\n");
fprintf("Network input size: ");
disp(inputSize);

%% ============================================================
% VALIDATION PREPROCESSING
% ============================================================

%
% Same deterministic preprocessing used during training:
%
% RGB
% 224 x 224
% single precision
% values [0,1]
%
% No random augmentation.
%

imdsValidation.ReadFcn = ...
    @prepareRFValidationImage;

%% ============================================================
% VALIDATION DATASTORE
% ============================================================

augmentedValidation = augmentedImageDatastore( ...
    inputSize(1:2), ...
    imdsValidation);

%% ============================================================
% CLASSIFY VALIDATION DATA
% ============================================================

disp(" ");
disp("====================================");
disp("CLASSIFYING VALIDATION DATA");
disp("====================================");

reset(augmentedValidation);

[YPredRaw, validationScores] = ...
    classify( ...
        net, ...
        augmentedValidation);

%% ============================================================
% GET TRUE LABELS
% ============================================================

YTrueRaw = ...
    imdsValidation.Labels;

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
% VERIFY DATA ALIGNMENT
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
    size(validationScores,1));

if numel(YTrue) ~= numel(YPred)

    error( ...
        "Ground-truth labels and predictions have different lengths.");

end

if size(validationScores,1) ~= numel(YTrue)

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

numClasses = ...
    numel(classes);

precision = ...
    zeros(numClasses,1);

recall = ...
    zeros(numClasses,1);

f1Score = ...
    zeros(numClasses,1);

support = ...
    zeros(numClasses,1);

for i = 1:numClasses

    %% ========================================================
    % TRUE POSITIVE
    % ========================================================

    truePositive = ...
        confusionMatrix(i,i);

    %% ========================================================
    % FALSE POSITIVE
    % ========================================================

    falsePositive = ...
        sum(confusionMatrix(:,i)) - ...
        truePositive;

    %% ========================================================
    % FALSE NEGATIVE
    % ========================================================

    falseNegative = ...
        sum(confusionMatrix(i,:)) - ...
        truePositive;

    %% ========================================================
    % SUPPORT
    % ========================================================

    support(i) = ...
        sum(confusionMatrix(i,:));

    %% ========================================================
    % PRECISION
    % ========================================================

    if ...
            (truePositive + falsePositive) > 0

        precision(i) = ...
            truePositive / ...
            (truePositive + falsePositive);

    else

        precision(i) = 0;

    end

    %% ========================================================
    % RECALL
    % ========================================================

    if ...
            (truePositive + falseNegative) > 0

        recall(i) = ...
            truePositive / ...
            (truePositive + falseNegative);

    else

        recall(i) = 0;

    end

    %% ========================================================
    % F1 SCORE
    % ========================================================

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
% FORCE COLUMN VECTORS
% ============================================================

classesColumn = ...
    string(classes(:));

support = ...
    double(support(:));

precision = ...
    double(precision(:));

recall = ...
    double(recall(:));

f1Score = ...
    double(f1Score(:));

%% ============================================================
% PERCENTAGES
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
% DISPLAY RESULTS
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
% METRICS TABLE
% ============================================================

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
% SAVE METRICS CSV
% ============================================================

metricsCSVPath = ...
    fullfile( ...
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
    'VariableNames', { ...
        'OverallAccuracy', ...
        'MacroPrecision', ...
        'MacroRecall', ...
        'MacroF1'});

summaryCSVPath = ...
    fullfile( ...
        resultsFolder, ...
        "validation_summary.csv");

writetable( ...
    summaryTable, ...
    summaryCSVPath);

%% ============================================================
% SAVE CONFUSION MATRIX CSV
% ============================================================

confusionCSVPath = ...
    fullfile( ...
        resultsFolder, ...
        "confusion_matrix.csv");

confusionTable = ...
    array2table( ...
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

confusionMATPath = ...
    fullfile( ...
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

summaryMATPath = ...
    fullfile( ...
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
% SAVE RAW VALIDATION SCORES
% ============================================================

scoresMATPath = ...
    fullfile( ...
        resultsFolder, ...
        "validation_scores.mat");

save( ...
    scoresMATPath, ...
    "validationScores", ...
    "YTrue", ...
    "YPred", ...
    "classes");

%% ============================================================
% FINAL OUTPUT
% ============================================================

fprintf("\n");
fprintf("====================================\n");
fprintf("EVALUATION COMPLETE\n");
fprintf("====================================\n\n");

fprintf( ...
    "Overall Accuracy: %.2f%%\n\n", ...
    overallAccuracyPercent);

fprintf( ...
    "Metrics CSV:\n%s\n\n", ...
    metricsCSVPath);

fprintf( ...
    "Confusion Matrix CSV:\n%s\n\n", ...
    confusionCSVPath);

fprintf( ...
    "Confusion Matrix MAT:\n%s\n\n", ...
    confusionMATPath);

fprintf( ...
    "Validation Summary CSV:\n%s\n\n", ...
    summaryCSVPath);

fprintf( ...
    "Validation Summary MAT:\n%s\n\n", ...
    summaryMATPath);

fprintf( ...
    "Validation Scores MAT:\n%s\n\n", ...
    scoresMATPath);

disp("No graphics were generated.");
disp("Evaluation completed without figure/export operations.");

end


%% ============================================================
% LOCAL FUNCTION:
% DETERMINISTIC VALIDATION IMAGE PREPROCESSING
% ============================================================

function I = prepareRFValidationImage(filename)

%% ============================================================
% READ IMAGE
% ============================================================

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

elseif size(I,3) == 4

    %% RGBA -> RGB

    I = I(:,:,1:3);

elseif size(I,3) ~= 3

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

if ~isequal( ...
        size(I), ...
        [224 224 3])

    error( ...
        "Validation preprocessing failed. Expected 224 x 224 x 3.");

end

end