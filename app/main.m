function main()

    % RF Signal Classification GUI

    %% Create GUI
    fig = uifigure( ...
        'Name', 'RF Signal Classification', ...
        'Position', [100 100 1000 650]);

    %% Title
    uilabel(fig, ...
        'Text', 'RF SIGNAL CLASSIFICATION', ...
        'FontSize', 24, ...
        'FontWeight', 'bold', ...
        'Position', [300 590 400 40]);

    %% Image display
    ax = uiaxes(fig, ...
        'Position', [50 220 420 330]);

    title(ax, 'RF Signal Image');
    ax.XTick = [];
    ax.YTick = [];

    %% Prediction panel
    panel = uipanel(fig, ...
        'Title', 'Prediction', ...
        'FontSize', 16, ...
        'Position', [520 330 420 220]);

    predictionLabel = uilabel(panel, ...
        'Text', 'Prediction: --', ...
        'FontSize', 22, ...
        'FontWeight', 'bold', ...
        'Position', [30 120 350 40]);

    confidenceLabel = uilabel(panel, ...
        'Text', 'Confidence: --', ...
        'FontSize', 18, ...
        'Position', [30 70 350 35]);

    %% Top predictions
    topPanel = uipanel(fig, ...
        'Title', 'Top 3 Predictions', ...
        'FontSize', 16, ...
        'Position', [520 50 420 250]);

    topText = uitextarea(topPanel, ...
        'Position', [20 20 380 180], ...
        'Editable', 'off', ...
        'FontSize', 15);

    %% Upload button
    uibutton(fig, ...
        'Text', 'Upload Image', ...
        'FontSize', 16, ...
        'Position', [80 130 180 50], ...
        'ButtonPushedFcn', @uploadImage);

    %% Classify button
    uibutton(fig, ...
        'Text', 'Classify', ...
        'FontSize', 16, ...
        'Position', [280 130 180 50], ...
        'ButtonPushedFcn', @classifyImage);

    %% Reset button
    uibutton(fig, ...
        'Text', 'Reset', ...
        'FontSize', 16, ...
        'Position', [520 10 120 35], ...
        'ButtonPushedFcn', @resetGUI);

    %% Store selected image
    selectedImage = '';

    %% Upload image
    function uploadImage(~, ~)

        [file, path] = uigetfile( ...
            {'*.png;*.jpg;*.jpeg', 'Image Files'});

        if isequal(file, 0)
            return;
        end

        selectedImage = fullfile(path, file);

        img = imread(selectedImage);

        imshow(img, 'Parent', ax);

        predictionLabel.Text = 'Prediction: Ready';
        confidenceLabel.Text = 'Confidence: --';

        topText.Value = {'Image loaded.', ...
                         'Click Classify.'};
    end

    %% Classify image
    function classifyImage(~, ~)

        if isempty(selectedImage)

            uialert(fig, ...
                'Please upload an RF signal image first.', ...
                'No Image');

            return;
        end

        try

            %% Load trained model
            net = loadRFModel();

            %% Classify image
            % Correct function signature:
            % classifyRFImage(imageInput, net)

            [predictedClass, confidence, probabilities] = ...
                classifyRFImage(selectedImage, net);

            %% Get Top 3 Predictions
            % Correct function signature:
            % getTopPredictions(scores, net, numberOfPredictions)

            topPredictions = ...
                getTopPredictions(probabilities, net, 3);

            %% Display prediction

            predictionLabel.Text = ...
                ['Prediction: ' char(predictedClass)];

            confidenceLabel.Text = ...
                sprintf('Confidence: %.2f%%', confidence);

            %% Display Top 3 Predictions

            lines = strings(height(topPredictions), 1);

            for i = 1:height(topPredictions)

                lines(i) = sprintf( ...
                    '%d. %s    %.2f%%', ...
                    i, ...
                    topPredictions.Class(i), ...
                    topPredictions.Confidence(i));

            end

            topText.Value = cellstr(lines);

        catch ME

            uialert(fig, ...
                ME.message, ...
                'Classification Error');

        end
    end

    %% Reset
    function resetGUI(~, ~)

        cla(ax);

        title(ax, 'RF Signal Image');

        selectedImage = '';

        predictionLabel.Text = 'Prediction: --';
        confidenceLabel.Text = 'Confidence: --';

        topText.Value = {};

    end

end