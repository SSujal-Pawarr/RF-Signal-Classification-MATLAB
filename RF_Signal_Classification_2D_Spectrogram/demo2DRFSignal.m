function demo2DRFSignal()
%% ============================================================
%  INTERACTIVE 2-D RF SIGNAL CLASSIFICATION DEMO
%  WLAN vs Bluetooth Low Energy vs LoRa vs FM
%
%  Input:
%       Raw I/Q samples
%
%  Processing:
%       I/Q -> Complex Signal -> Spectrogram -> Normalization
%
%  Model:
%       Trained 2-D CNN
%
%  Interface:
%       MATLAB Command Window + file selection
%
%  MATLAB R2026b
%% ============================================================

clc;
close all;

%% ============================================================
%  PROJECT PATHS
%% ============================================================

projectPath = fileparts(mfilename('fullpath'));

addpath(genpath(projectPath));

dataFile = fullfile( ...
    projectPath, ...
    'data', ...
    'spectrogram', ...
    'spectrogramWirelessDataset.mat');

modelFile = fullfile( ...
    projectPath, ...
    'models', ...
    'wirelessIQ2DClassifier.mat');

%% ============================================================
%  CHECK FILES
%% ============================================================

if ~isfile(dataFile)
    error(['Dataset not found.\n' ...
           'Run runProject2D first to generate the dataset.']);
end

if ~isfile(modelFile)
    error(['Trained 2-D CNN model not found.\n' ...
           'Run runProject2D first to train the model.']);
end

%% ============================================================
%  LOAD DATASET AND MODEL
%% ============================================================

load(dataFile, 'dataset');
load(modelFile, 'net', 'classNames');

classNames = string(classNames(:));

%% ============================================================
%  DISPLAY MENU
%% ============================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf('       FOUR-CLASS 2-D RF CLASSIFICATION DEMO\n');
fprintf('====================================================\n');

fprintf('\nAvailable input options:\n\n');

fprintf('1. WLAN test sample\n');
fprintf('2. BluetoothLE test sample\n');
fprintf('3. LoRa test sample\n');
fprintf('4. FM test sample\n');
fprintf('5. Random test sample\n');
fprintf('6. Custom I/Q MAT file\n');
fprintf('7. Evaluate complete test set\n');
fprintf('8. Exit\n');

choice = input('\nEnter choice (1-8): ');

%% ============================================================
%  SELECT INPUT
%% ============================================================

if choice >= 1 && choice <= 4

    % --------------------------------------------------------
    % Known simulated class
    % --------------------------------------------------------

    label = classNames(choice);

    iqComplex = simulateRFSignal( ...
        label, ...
        dataset.N, ...
        dataset.fs);

    iq = [real(iqComplex), imag(iqComplex)];

    source = "Simulated test-style sample: " + label;

    knownLabel = string(label);

elseif choice == 5

    % --------------------------------------------------------
    % Random simulated class
    % --------------------------------------------------------

    label = classNames(randi(numel(classNames)));

    iqComplex = simulateRFSignal( ...
        label, ...
        dataset.N, ...
        dataset.fs);

    iq = [real(iqComplex), imag(iqComplex)];

    source = "Random simulated sample";

    knownLabel = string(label);

elseif choice == 6

    % --------------------------------------------------------
    % Custom I/Q MAT file
    % --------------------------------------------------------

    [fileName, filePath] = uigetfile( ...
        '*.mat', ...
        'Select custom I/Q MAT file');

    if isequal(fileName, 0)

        fprintf('\nCustom file selection cancelled.\n');

        return;

    end

    S = load(fullfile(filePath, fileName));

    iq = extractIQ(S);

    % If input is 2 x N, convert it to N x 2
    if size(iq,2) ~= 2 && size(iq,1) == 2
        iq = iq.';
    end

    % Validate input
    if size(iq,2) ~= 2
        error(['Custom I/Q data must contain exactly two columns:\n' ...
               'Column 1 = I\n' ...
               'Column 2 = Q']);
    end

    % Convert to double
    iq = double(iq);

    source = "Custom MAT file: " + string(fileName);

    knownLabel = "Unknown";

elseif choice == 7

    % --------------------------------------------------------
    % Evaluate complete test set
    % --------------------------------------------------------

    evaluate2DRFClassifier();

    return;

elseif choice == 8

    fprintf('\nDemo closed.\n');

    return;

else

    error('Invalid choice. Please enter a number from 1 to 8.');

end

%% ============================================================
%  DISPLAY INPUT INFORMATION
%% ============================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf('                 INPUT INFORMATION\n');
fprintf('====================================================\n');

fprintf('Input source : %s\n', source);
fprintf('Input samples: %d\n', size(iq,1));
fprintf('Input columns: I and Q\n');

fprintf('====================================================\n');

%% ============================================================
%  CREATE COMPLEX I/Q SIGNAL
%% ============================================================

z = complex(iq(:,1), iq(:,2));

%% ============================================================
%  PREPROCESSING
%
%  Raw I/Q
%       ↓
%  Complex signal
%       ↓
%  Spectrogram
%       ↓
%  Log magnitude
%       ↓
%  Normalization
%% ============================================================

fprintf('\n');
fprintf('[1/4] Converting I/Q samples into complex signal...\n');

fprintf('[2/4] Generating time-frequency spectrogram...\n');

[S, F, T] = spectrogram( ...
    z, ...
    dataset.spectrogramWindow, ...
    dataset.spectrogramOverlap, ...
    dataset.nfft, ...
    dataset.fs, ...
    'centered');

% Magnitude spectrogram
magnitudeSpectrogram = abs(S);

% Log compression
img = single(log1p(magnitudeSpectrogram));

fprintf('[3/4] Normalizing spectrogram...\n');

% Use the same normalization values used during training
img = (img - dataset.normalizationMean) ...
          / dataset.normalizationStd;

%% ============================================================
%  PREPARE INPUT FOR 2-D CNN
%% ============================================================

X = reshape( ...
    img, ...
    size(img,1), ...
    size(img,2), ...
    1, ...
    1);

fprintf('[4/4] Preparing image for 2-D CNN...\n');

%% ============================================================
%  CLASSIFICATION
%% ============================================================

fprintf('\n');
fprintf('Running 2-D CNN classification...\n');

[predictedLabel, scores] = classify(net, X);

scores = scores(:);

%% ============================================================
%  COMMAND WINDOW OUTPUT
%% ============================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf('                 2-D CNN OUTPUT\n');
fprintf('====================================================\n');

fprintf('Input source: %s\n', source);

fprintf('Predicted class: %s\n', ...
    string(predictedLabel));

if knownLabel ~= "Unknown"

    fprintf('Known simulated label: %s\n', ...
        knownLabel);

    predictionCorrect = ...
        string(predictedLabel) == knownLabel;

    fprintf('Prediction correct: %s\n', ...
        string(predictionCorrect));

else

    fprintf(['Known label: Unknown ' ...
             '(custom input has no ground-truth label)\n']);

end

fprintf('\nClass scores:\n');

for k = 1:min(numel(classNames), numel(scores))

    fprintf('  %-15s %.4f\n', ...
        classNames(k), ...
        scores(k));

end

fprintf('====================================================\n');

%% ============================================================
%  FIGURE 1
%  RAW I/Q INPUT
%% ============================================================

figure( ...
    'Name', '2-D Input I/Q', ...
    'NumberTitle', 'off', ...
    'Color', 'w');

tiledlayout(2,1);

%% I COMPONENT

nexttile;

plot(iq(:,1), ...
     'LineWidth', 1);

title( ...
    'Input RF Signal - In-phase (I)', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

xlabel( ...
    'Sample Index', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

ylabel( ...
    'Amplitude', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

grid on;

ax = gca;

ax.XColor = 'k';
ax.YColor = 'k';

ax.FontWeight = 'bold';

%% Q COMPONENT

nexttile;

plot(iq(:,2), ...
     'LineWidth', 1);

title( ...
    'Input RF Signal - Quadrature (Q)', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

xlabel( ...
    'Sample Index', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

ylabel( ...
    'Amplitude', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

grid on;

ax = gca;

ax.XColor = 'k';
ax.YColor = 'k';

ax.FontWeight = 'bold';

%% ============================================================
%  FIGURE 2
%  SPECTROGRAM
%% ============================================================

figure( ...
    'Name', '2-D CNN Spectrogram Input', ...
    'NumberTitle', 'off', ...
    'Color', 'w');

% Display log-magnitude spectrogram
imagesc( ...
    T, ...
    F/1e3, ...
    log1p(magnitudeSpectrogram));

axis xy;

%% TITLE

title( ...
    '2-D Time-Frequency Representation (Spectrogram)', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

%% X AXIS

xlabel( ...
    'Time (s)', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

%% Y AXIS

ylabel( ...
    'Frequency (kHz)', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

%% COLORBAR

cb = colorbar;

cb.Label.String = 'Log Magnitude';

cb.Label.Color = 'k';

cb.Color = 'k';

%% AXIS COLOR

ax = gca;

ax.XColor = 'k';
ax.YColor = 'k';

ax.FontWeight = 'bold';

ax.Color = 'w';

%% ============================================================
%  FIGURE 3
%  CLASSIFICATION SCORES
%% ============================================================

figure( ...
    'Name', '2-D CNN Scores', ...
    'NumberTitle', 'off', ...
    'Color', 'w');

bar(scores);

xticks(1:numel(scores));

xticklabels(classNames);

xlabel( ...
    'RF Signal Class', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

ylabel( ...
    'Model Score', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

title( ...
    '2-D CNN Classification Scores', ...
    'Color', 'k', ...
    'FontWeight', 'bold');

ylim([0 1.05]);

grid on;

ax = gca;

ax.XColor = 'k';
ax.YColor = 'k';

ax.FontWeight = 'bold';

%% ============================================================
%  FINAL SUMMARY
%% ============================================================

fprintf('\n');
fprintf('====================================================\n');
fprintf('             DEMONSTRATION COMPLETED\n');
fprintf('====================================================\n');

fprintf('Input          : Raw I/Q signal samples\n');
fprintf('Representation : 2-D time-frequency spectrogram\n');
fprintf('AI model       : Trained 2-D CNN\n');
fprintf('Prediction     : %s\n', ...
    string(predictedLabel));

fprintf('\n');

fprintf('Processing flow:\n');
fprintf('Raw I/Q\n');
fprintf('   -> Complex signal\n');
fprintf('   -> Spectrogram\n');
fprintf('   -> Log magnitude\n');
fprintf('   -> Normalization\n');
fprintf('   -> 2-D CNN\n');
fprintf('   -> Predicted RF class\n');

fprintf('\n');

fprintf('Note: Results are based on simulated I/Q data.\n');
fprintf('They do not imply real-world RF performance.\n');

fprintf('====================================================\n');

end


%% ============================================================
%  FUNCTION: EXTRACT CUSTOM I/Q DATA
%% ============================================================

function iq = extractIQ(S)

names = ["iq", "IQ", "X", "rawIQ"];

for n = names

    if isfield(S,n)

        value = S.(n);

        if isnumeric(value) && ismatrix(value)

            iq = value;

            return;

        end

    end

end

error([ ...
    'MAT file must contain a numeric variable named ', ...
    'iq, IQ, X, or rawIQ.']);

end


%% ============================================================
%  FUNCTION: SIMULATE RF SIGNAL
%
%  Creates simulated I/Q data for:
%       WLAN
%       BluetoothLE
%       LoRa
%       FM
%
%  No hardware is required.
%% ============================================================

function z = simulateRFSignal(label,N,fs)

t = (0:N-1)' / fs;

switch string(label)

    %% ========================================================
    %  WLAN
    %% ========================================================

    case "WLAN"

        z = zeros(N,1);

        for k = 1:12

            f = (k - 6.5) * fs / 32;

            sym = randi( ...
                [0 3], ...
                1, ...
                ceil(N/64));

            sym = exp( ...
                1j * (pi/2) * sym);

            a = repelem( ...
                sym, ...
                64).';

            a = a(1:N);

            z = z + ...
                a .* ...
                exp(1j * 2*pi*f*t);

        end

        z = z + ...
            0.25 * ...
            (randn(N,1) + ...
             1j*randn(N,1));


    %% ========================================================
    %  BLUETOOTH LE
    %% ========================================================

    case "BluetoothLE"

        bits = randi( ...
            [0 1], ...
            ceil(N/32), ...
            1);

        nr = repelem( ...
            2*bits - 1, ...
            32);

        nr = nr(1:N);

        phase = 2*pi * ...
            cumsum(0.10 * fs * nr) / fs;

        z = exp(1j*phase);

        z = z + ...
            0.18 * ...
            (randn(N,1) + ...
             1j*randn(N,1));


    %% ========================================================
    %  LORA
    %% ========================================================

    case "LoRa"

        bw = 0.30 * fs;

        rate = bw / (N/fs);

        phase = 2*pi * ...
            ((-bw/2)*t + ...
             0.5*rate*t.^2);

        z = exp(1j*phase);

        z = z + ...
            0.16 * ...
            (randn(N,1) + ...
             1j*randn(N,1));


    %% ========================================================
    %  FM
    %% ========================================================

    case "FM"

        msg = ...
            sin(2*pi*0.015*fs*t) + ...
            0.35*sin(2*pi*0.00675*fs*t);

        z = exp( ...
            1j * ...
            1.8 * msg);

        z = z + ...
            0.20 * ...
            (randn(N,1) + ...
             1j*randn(N,1));


    %% ========================================================
    %  UNKNOWN
    %% ========================================================

    otherwise

        error('Unknown RF signal class.');

end

%% ============================================================
%  NORMALIZE SIGNAL
%% ============================================================

z = z ./ ...
    max(abs(z) + eps);

end