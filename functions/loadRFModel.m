function net = loadRFModel()

clc;

fprintf('\n');
fprintf('============================================================\n');
fprintf('       RF SIGNAL CLASSIFICATION - MODEL LOADER\n');
fprintf('============================================================\n\n');

%% Project path
projectRoot = fileparts(fileparts(mfilename('fullpath')));

%% Our locally trained MATLAB model
modelPath = fullfile( ...
    projectRoot, ...
    'models', ...
    'rfSignalNet.mat');

fprintf('Model path:\n%s\n\n', modelPath);

%% Check model
if ~isfile(modelPath)
    error( ...
        'Trained MATLAB model not found: %s', ...
        modelPath);
end

fprintf('Model file found successfully.\n\n');
fprintf('Loading trained MATLAB model...\n\n');

%% Load network
data = load(modelPath, 'net');

if ~isfield(data, 'net')
    error('The MAT file does not contain a variable named "net".');
end

net = data.net;

fprintf('============================================================\n');
fprintf('MODEL LOADED SUCCESSFULLY\n');
fprintf('============================================================\n\n');

fprintf('Network input size: ');
disp(net.Layers(1).InputSize);

end