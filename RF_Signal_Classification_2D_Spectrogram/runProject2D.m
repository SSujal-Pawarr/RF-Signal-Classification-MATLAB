%% 2-D RF SIGNAL CLASSIFICATION PROJECT
clc; clear; close all;
projectPath = fileparts(mfilename('fullpath'));
addpath(genpath(projectPath));


fprintf('\n====================================================\n');
fprintf('       FOUR-CLASS 2-D RF CLASSIFICATION\n');
fprintf('          SPECTROGRAM + 2-D CNN\n');
fprintf('====================================================\n');

fprintf('\n[1/3] Generating simulated I/Q and spectrogram dataset...\n');
generate2DSpectrogramDataset();

fprintf('\n[2/3] Training 2-D CNN...\n');
train2DRFClassifier();

fprintf('\n[3/3] Evaluating held-out test set...\n');
evaluate2DRFClassifier();

fprintf('\n====================================================\n');
fprintf('          2-D PROJECT READY\n');
fprintf('====================================================\n');
fprintf('Run demo2DRFSignal for the interactive Command Window demo.\n');
fprintf('====================================================\n');
