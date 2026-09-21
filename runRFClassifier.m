%% RF SIGNAL CLASSIFICATION - MAIN PROGRAM

clc;
clear;

fprintf('\n');
fprintf('============================================================\n');
fprintf('          RF SIGNAL CLASSIFICATION SYSTEM\n');
fprintf('============================================================\n\n');

%% 1. Project setup

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot,'functions'));

%% 2. Load our locally trained model

model = loadRFModel();

%% 3. Select an RF signal image

[fileName, filePath] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg','Image Files (*.png, *.jpg, *.jpeg)'}, ...
    'Select an RF Signal Image');

if isequal(fileName,0)
    fprintf('No image selected.\n');
    return;
end

imagePath = fullfile(filePath,fileName);

fprintf('\nSelected image:\n%s\n\n',imagePath);

%% 4. Classify image

[predictedLabel, confidence, scores] = ...
    classifyRFImage(imagePath,model);

%% 5. Display result

fprintf('============================================================\n');
fprintf('                  CLASSIFICATION RESULT\n');
fprintf('============================================================\n');

fprintf('Image      : %s\n',fileName);
fprintf('Signal     : %s\n',string(predictedLabel));
fprintf('Confidence : %.2f%%\n',confidence*100);

fprintf('============================================================\n\n');

%% 6. Get top 3 predictions

topPredictions = getTopPredictions(scores,model);

fprintf('TOP PREDICTIONS\n');
fprintf('------------------------------------------------------------\n');

disp(topPredictions);

%% 7. Display selected image

figure('Name','RF Signal Classification');

imshow(imread(imagePath));

title(sprintf( ...
    'Predicted: %s | Confidence: %.2f%%', ...
    string(predictedLabel), ...
    confidence*100));

fprintf('\nClassification completed successfully.\n');