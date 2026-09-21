function threshold = calibrateUnknownThreshold(net)
% CALIBRATEUNKNOWNTHRESHOLD
% Calculates a confidence threshold for UNKNOWN detection
% using the existing validation dataset.
%
% The trained RF model is NOT changed.
%
% Output:
%   threshold - confidence threshold between 0 and 1

clc;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       UNKNOWN THRESHOLD CALIBRATION\n');
fprintf('============================================================\n\n');

%% Project paths

projectRoot = fileparts(fileparts(mfilename('fullpath')));

validationFolder = fullfile( ...
    projectRoot, ...
    'data', ...
    'dataset', ...
    'validation');

resultsFolder = fullfile(projectRoot,'results');

if ~isfolder(validationFolder)
    error('Validation folder not found: %s', validationFolder);
end

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

%% Class names

classNames = categories(net.Layers(end).Classes);

fprintf('Validation folder:\n%s\n\n', validationFolder);

fprintf('Classes:\n');
disp(classNames);

%% Load validation images

imdsValidation = imageDatastore( ...
    validationFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('Total validation images: %d\n\n', ...
    numel(imdsValidation.Files));

%% Storage

numImages = numel(imdsValidation.Files);

maxConfidence = zeros(numImages,1);
predictedLabels = strings(numImages,1);
trueLabels = string(imdsValidation.Labels);

%% Evaluate every validation image

fprintf('Calculating confidence scores...\n\n');

for i = 1:numImages

    imagePath = imdsValidation.Files{i};

    img = imread(imagePath);

    % Same deterministic preprocessing used during validation
    img = prepareValidationImage(img);

    [predictedLabel, scores] = classify(net,img);

    scores = double(scores);

    % Ensure scores are treated as probabilities
    if any(scores < 0) || abs(sum(scores) - 1) > 1e-3
        error(['Unexpected score output detected. ' ...
               'Scores are not valid probabilities.']);
    end

    maxConfidence(i) = max(scores);

    predictedLabels(i) = string(predictedLabel);

    if mod(i,50) == 0 || i == numImages
        fprintf('Processed %d / %d images\n', ...
            i,numImages);
    end
end

%% Correct predictions

isCorrect = predictedLabels == trueLabels;

correctConfidence = maxConfidence(isCorrect);

incorrectConfidence = maxConfidence(~isCorrect);

numCorrect = sum(isCorrect);
numIncorrect = sum(~isCorrect);

fprintf('\n');
fprintf('============================================================\n');
fprintf('VALIDATION CONFIDENCE SUMMARY\n');
fprintf('============================================================\n\n');

fprintf('Correct predictions   : %d\n',numCorrect);
fprintf('Incorrect predictions : %d\n',numIncorrect);

fprintf('\n');

fprintf('Correct prediction confidence:\n');
fprintf('  Minimum : %.4f\n',min(correctConfidence));
fprintf('  Mean    : %.4f\n',mean(correctConfidence));
fprintf('  Median  : %.4f\n',median(correctConfidence));
fprintf('  Maximum : %.4f\n',max(correctConfidence));

if ~isempty(incorrectConfidence)

    fprintf('\n');

    fprintf('Incorrect prediction confidence:\n');
    fprintf('  Minimum : %.4f\n',min(incorrectConfidence));
    fprintf('  Mean    : %.4f\n',mean(incorrectConfidence));
    fprintf('  Median  : %.4f\n',median(incorrectConfidence));
    fprintf('  Maximum : %.4f\n',max(incorrectConfidence));

end

%% Select threshold

% We want the threshold to be below the confidence
% of almost all correct validation predictions.

sortedCorrect = sort(correctConfidence);

% Use the 5th percentile of correct predictions.
threshold = prctile(sortedCorrect,5);

% Keep threshold inside a sensible range.
threshold = max(0.50,min(0.95,threshold));

fprintf('\n');
fprintf('============================================================\n');
fprintf('SELECTED UNKNOWN THRESHOLD\n');
fprintf('============================================================\n\n');

fprintf('Threshold = %.4f (%.2f%%)\n', ...
    threshold,threshold*100);

fprintf('\nInterpretation:\n');
fprintf('Confidence >= threshold -> KNOWN class\n');
fprintf('Confidence <  threshold -> UNKNOWN\n');

%% Save threshold

thresholdPath = fullfile( ...
    resultsFolder, ...
    'unknown_threshold.mat');

save(thresholdPath,'threshold');

fprintf('\nThreshold saved to:\n%s\n',thresholdPath);

%% Save detailed calibration data

calibrationData = table( ...
    trueLabels, ...
    predictedLabels, ...
    maxConfidence, ...
    isCorrect, ...
    'VariableNames', { ...
    'TrueLabel', ...
    'PredictedLabel', ...
    'Confidence', ...
    'Correct'});

csvPath = fullfile( ...
    resultsFolder, ...
    'unknown_threshold_calibration.csv');

writetable(calibrationData,csvPath);

fprintf('\nCalibration data saved to:\n%s\n',csvPath);

fprintf('\n');
fprintf('============================================================\n');
fprintf('CALIBRATION COMPLETE\n');
fprintf('============================================================\n\n');

end


%% ============================================================
% Local validation preprocessing
% ============================================================

function img = prepareValidationImage(img)

    if ndims(img) == 2

        img = cat(3,img,img,img);

    elseif size(img,3) == 4

        img = img(:,:,1:3);

    elseif size(img,3) ~= 3

        error('Unexpected image channel count.');

    end

    img = im2single(img);

    img = imresize(img,[224 224]);

end