function trainWirelessClassifier()
root=fileparts(mfilename('fullpath')); dataFile=fullfile(root,'data','labeled_iq','labeledWirelessIQDataset.mat');
modelFile=fullfile(root,'models','wirelessIQClassifier.mat'); load(dataFile,'dataset');
classes=string(categories(dataset.YTrain)); nc=numel(classes);
layers=[sequenceInputLayer(2,'MinLength',dataset.N)
 convolution1dLayer(7,32,'Padding','same'); batchNormalizationLayer; reluLayer
 maxPooling1dLayer(2,'Stride',2)
 convolution1dLayer(5,64,'Padding','same'); batchNormalizationLayer; reluLayer
 maxPooling1dLayer(2,'Stride',2)
 convolution1dLayer(5,96,'Padding','same'); reluLayer
 globalAveragePooling1dLayer; fullyConnectedLayer(nc); softmaxLayer; classificationLayer];
opts=trainingOptions('adam','MaxEpochs',15,'MiniBatchSize',32,'InitialLearnRate',1e-3,'Shuffle','every-epoch',...
 'ValidationData',{dataset.XValidation,dataset.YValidation},'ValidationFrequency',20,'Verbose',true,'Plots','training-progress');
net=trainNetwork(dataset.XTrain,dataset.YTrain,layers,opts); classNames=classes;
save(modelFile,'net','classNames','-v7.3'); fprintf('\nSaved trained model: %s\n',modelFile);
end
