function inputImage = preprocessRFImage(imageInput)
% PREPROCESSRFIMAGE
% Preprocesses an RF spectrogram using the SAME image
% representation used during training.
%
% Output:
%   224 x 224 x 3
%   single precision
%   values in [0,1]

%% ============================================================
% READ IMAGE
% ============================================================

if ischar(imageInput) || isstring(imageInput)

    if ~isfile(imageInput)

        error( ...
            "Image file does not exist: %s", ...
            imageInput);

    end

    inputImage = imread(imageInput);

else

    inputImage = imageInput;

end

%% ============================================================
% CONVERT TO RGB
% ============================================================

if ndims(inputImage) == 2

    inputImage = cat( ...
        3, ...
        inputImage, ...
        inputImage, ...
        inputImage);

elseif size(inputImage, 3) == 4

    inputImage = ...
        inputImage(:, :, 1:3);

elseif size(inputImage, 3) ~= 3

    error( ...
        "Input image must have 1, 3, or 4 channels.");

end

%% ============================================================
% CONVERT TO SINGLE [0,1]
% ============================================================

inputImage = im2single(inputImage);

%% ============================================================
% RESIZE
% ============================================================

inputImage = imresize( ...
    inputImage, ...
    [224 224]);

%% ============================================================
% VERIFY
% ============================================================

if ~isequal(size(inputImage), [224 224 3])

    error( ...
        "Preprocessing failed. Expected 224 x 224 x 3.");

end

%% ============================================================
% DISPLAY INFORMATION
% ============================================================

% No normalization to [-1,1] is performed.
% The trained model uses single precision [0,1].

end