function net=train2DRFClassifier()
%% Train the 2-D CNN

projectPath=fileparts(mfilename('fullpath'));
dataFile=fullfile(projectPath,'data','spectrogram','spectrogramWirelessDataset.mat');
modelDir=fullfile(projectPath,'models');
if ~isfolder(modelDir), mkdir(modelDir); end

load(dataFile,'dataset');

inputSize=size(dataset.XTrain(:,:,:,1));
numClasses=numel(dataset.classNames);

layers=[
    imageInputLayer(inputSize,'Normalization','none','Name','input')
    convolution2dLayer(3,16,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')
    convolution2dLayer(3,32,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')
    convolution2dLayer(3,64,'Padding','same','Name','conv3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')
    globalAveragePooling2dLayer('Name','gap')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
];

opts=trainingOptions('adam',...
    'MaxEpochs',15,...
    'MiniBatchSize',32,...
    'InitialLearnRate',1e-3,...
    'Shuffle','every-epoch',...
    'ValidationData',{dataset.XValidation,dataset.YValidation},...
    'ValidationFrequency',max(1,floor(numel(dataset.YTrain)/32)),...
    'Verbose',true,...
    'Plots','training-progress');

net=trainNetwork(dataset.XTrain,dataset.YTrain,layers,opts);

classNames=dataset.classNames;
save(fullfile(modelDir,'wirelessIQ2DClassifier.mat'),'net','classNames','-v7.3');
fprintf('Saved 2-D CNN model.\n');
end
