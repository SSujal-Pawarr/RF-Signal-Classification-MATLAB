function [predictedLabel, confidence, scores] = classifyRFImage(imageInput, net)
% CLASSIFYRFIMAGE
% Classifies an RF spectrogram and supports UNKNOWN detection.
%
% Inputs:
%   imageInput - image file path or image matrix
%   net        - trained MATLAB classification network
%
% Outputs:
%   predictedLabel - predicted RF class or "UNKNOWN"
%   confidence     - highest model confidence
%   scores         - probability scores for all trained classes
%
% UNKNOWN detection:
%   The threshold is loaded from:
%       results/unknown_threshold.mat
%
%   The threshold must be calibrated beforehand using
%   calibrateUnknownThreshold.m.
%
%   No hardcoded UNKNOWN threshold is used.

%% ---------------------------------------------------------
% 1. Read input image
% ----------------------------------------------------------

if ischar(imageInput) || isstring(imageInput)

    if ~isfile(imageInput)
        error("Image file does not exist: %s", imageInput);
    end

    inputImage = imread(imageInput);

else

    inputImage = imageInput;

end


%% ---------------------------------------------------------
% 2. Preprocess image
% ----------------------------------------------------------

inputImage = preprocessRFImage(inputImage);


%% ---------------------------------------------------------
% 3. Classify image
% ----------------------------------------------------------

[predictedLabel, scores] = classify(net, inputImage);

confidence = max(scores);


%% ---------------------------------------------------------
% 4. Locate UNKNOWN threshold file
% ----------------------------------------------------------

projectRoot = fileparts(fileparts(mfilename('fullpath')));

thresholdPath = fullfile( ...
    projectRoot, ...
    'results', ...
    'unknown_threshold.mat');


%% ---------------------------------------------------------
% 5. Require calibrated threshold
% ----------------------------------------------------------

if ~isfile(thresholdPath)

    error([ ...
        'UNKNOWN threshold file not found: %s\n' ...
        'Run calibrateUnknownThreshold before using UNKNOWN detection.' ...
        ], thresholdPath);

end


%% ---------------------------------------------------------
% 6. Load calibrated threshold
% ----------------------------------------------------------

data = load(thresholdPath, 'threshold');


if ~isfield(data, 'threshold')

    error( ...
        'unknown_threshold.mat does not contain "threshold".');

end


unknownThreshold = data.threshold;


%% ---------------------------------------------------------
% 7. Validate threshold
% ----------------------------------------------------------

if ~isscalar(unknownThreshold) || ...
        ~isnumeric(unknownThreshold) || ...
        ~isfinite(unknownThreshold)

    error( ...
        'Invalid UNKNOWN threshold in unknown_threshold.mat.');

end


if unknownThreshold < 0 || unknownThreshold > 1

    error( ...
        'UNKNOWN threshold must be between 0 and 1. Current value: %.6f', ...
        unknownThreshold);

end


%% ---------------------------------------------------------
% 8. UNKNOWN decision
% ----------------------------------------------------------

if confidence < unknownThreshold

    predictedLabel = categorical( ...
        "UNKNOWN", ...
        ["UNKNOWN"; string(net.Layers(end).Classes)]);

end

end
