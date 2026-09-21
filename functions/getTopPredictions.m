function topPredictions = getTopPredictions(scores, net, numberOfPredictions)
% getTopPredictions
% Returns the highest scoring RF signal classes.
%
% Usage:
%   topPredictions = getTopPredictions(scores, net)
%   topPredictions = getTopPredictions(scores, net, 3)

%% Default number of predictions
if nargin < 3
    numberOfPredictions = 3;
end

%% Convert scores to a normal numeric vector
scores = double(scores(:));

%% Get class names from the trained network
outputLayer = net.Layers(end);

if isprop(outputLayer, "Classes")
    classNames = string(outputLayer.Classes);
else
    % Fallback for the current network
    classNames = string( ...
        ["fm","bluetooth","wifi","cellular", ...
        "lora","ais","airband","RS41-Radiosonde"]);
end

%% Sort scores from highest to lowest
[sortedScores, sortedIndices] = sort( ...
    scores, "descend");

%% Limit number of predictions
numberOfPredictions = min( ...
    double(numberOfPredictions), ...
    numel(sortedScores));

%% Create result table
topPredictions = table( ...
    classNames(sortedIndices(1:numberOfPredictions)), ...
    sortedScores(1:numberOfPredictions) * 100, ...
    'VariableNames', {'Signal','Confidence'});

end