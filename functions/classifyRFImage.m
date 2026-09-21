function [predictedLabel, confidence, scores] = classifyRFImage(imageInput, net)
% CLASSIFYRFIMAGE
% Classifies an RF spectrogram and supports UNKNOWN detection.
%
% Output:
%   predictedLabel - predicted RF class or "UNKNOWN"
%   confidence     - highest model confidence
%   scores         - probability scores for all classes

%% Load image

if ischar(imageInput) || isstring(imageInput)

    if ~isfile(imageInput)
        error("Image file does not exist: %s", imageInput);
    end

    inputImage = imread(imageInput);

else

    inputImage = imageInput;

end

%% Preprocess

inputImage = preprocessRFImage(inputImage);

%% Classification

[predictedLabel, scores] = classify(net,inputImage);

%% Convert scores to double

scores = double(scores);

%% Get confidence

confidence = max(scores);

%% Load calibrated UNKNOWN threshold

projectRoot = fileparts(fileparts(mfilename('fullpath')));

thresholdPath = fullfile( ...
    projectRoot, ...
    'results', ...
    'unknown_threshold.mat');

if isfile(thresholdPath)

    data = load(thresholdPath,'threshold');

    if ~isfield(data,'threshold')
        error('unknown_threshold.mat does not contain "threshold".');
    end

    unknownThreshold = data.threshold;

else

    % Safe fallback if calibration file does not exist
    unknownThreshold = 0.9305;

end

%% UNKNOWN detection

if confidence < unknownThreshold

    predictedLabel = categorical( ...
        "UNKNOWN", ...
        ["UNKNOWN"; string(net.Layers(end).Classes)]);

end

end