function visualizeRFSignal(imageInput, predictedLabel, confidence, topPredictions)
% visualizeRFSignal
% Displays the RF image, predicted signal, confidence,
% and top predictions.

%% Read image
if ischar(imageInput) || isstring(imageInput)
    inputImage = imread(imageInput);
else
    inputImage = imageInput;
end

%% Create figure
figure('Name','RF Signal Classification', ...
    'NumberTitle','off');

%% Display image
subplot(1,2,1);

imshow(inputImage);
title(sprintf('Predicted: %s\nConfidence: %.2f%%', ...
    string(predictedLabel), confidence*100), ...
    'Interpreter','none');

%% Display top predictions
subplot(1,2,2);

bar(topPredictions.Confidence);

xticks(1:height(topPredictions));
xticklabels(topPredictions.Signal);

xtickangle(45);

ylabel('Confidence (%)');
xlabel('RF Signal');
title('Top Predictions');

grid on;

end