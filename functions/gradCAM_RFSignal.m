function [scoreMap, predictedLabel, confidence] = ...
    gradCAM_RFSignal(imageInput, net)
% GRADCAM_RFSIGNAL
% Generates a Grad-CAM heatmap for an RF signal classification.
%
% Input:
%   imageInput - image path or image array
%   net        - trained RF classification network
%
% Output:
%   scoreMap       - Grad-CAM heatmap
%   predictedLabel - predicted RF class
%   confidence     - prediction confidence

%% Check inputs

if nargin < 2
    error("Usage: gradCAM_RFSignal(imageInput, net)");
end

%% Load image

if ischar(imageInput) || isstring(imageInput)

    if ~isfile(imageInput)
        error("Image file does not exist: %s", imageInput);
    end

    originalImage = imread(imageInput);

else

    originalImage = imageInput;

end

%% Preprocess image

inputImage = preprocessRFImage(originalImage);

%% Get prediction

[predictedLabel, scores] = classify(net,inputImage);

scores = double(scores);

confidence = max(scores);

fprintf('\n');
fprintf('============================================================\n');
fprintf('                 GRAD-CAM ANALYSIS\n');
fprintf('============================================================\n\n');

fprintf('Predicted class : %s\n',string(predictedLabel));
fprintf('Confidence      : %.2f%%\n',confidence*100);

%% Check for UNKNOWN

if string(predictedLabel) == "UNKNOWN"

    fprintf('\n');
    fprintf('The image was rejected as UNKNOWN.\n');
    fprintf('Grad-CAM will not be generated for UNKNOWN.\n\n');

    scoreMap = [];

    return;

end

%% Generate Grad-CAM

try

    scoreMap = gradCAM( ...
        net, ...
        inputImage, ...
        predictedLabel, ...
        "ReductionLayer", "rf_softmax", ...
        "OutputUpsampling", "bicubic");

catch ME

    fprintf('\nGrad-CAM failed using the current DAGNetwork.\n');
    fprintf('MATLAB error:\n%s\n\n',ME.message);

    fprintf('The model may need to be converted to dlnetwork.\n');

    rethrow(ME);

end

%% Normalize score map

scoreMap = double(scoreMap);

minValue = min(scoreMap(:));
maxValue = max(scoreMap(:));

if maxValue > minValue

    scoreMap = ...
        (scoreMap - minValue) ./ ...
        (maxValue - minValue);

else

    scoreMap = zeros(size(scoreMap));

end

%% Display summary

fprintf('\n');
fprintf('Grad-CAM generated successfully.\n');
fprintf('Heatmap size: %d x %d\n', ...
    size(scoreMap,1),size(scoreMap,2));

fprintf('\n');
fprintf('============================================================\n');
fprintf('GRAD-CAM COMPLETE\n');
fprintf('============================================================\n\n');

end