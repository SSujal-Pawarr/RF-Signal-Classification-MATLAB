function accuracy=evaluate2DRFClassifier()
%% Evaluate held-out test data

projectPath=fileparts(mfilename('fullpath'));
load(fullfile(projectPath,'data','spectrogram','spectrogramWirelessDataset.mat'),'dataset');
load(fullfile(projectPath,'models','wirelessIQ2DClassifier.mat'),'net','classNames');

predictedLabels=classify(net,dataset.XTest);
trueLabels=dataset.YTest;
accuracy=mean(predictedLabels==trueLabels)*100;
correct=sum(predictedLabels==trueLabels);
total=numel(trueLabels);

fprintf('\n====================================================\n');
fprintf('              2-D CNN TEST RESULTS\n');
fprintf('====================================================\n');
fprintf('Total test signals    : %d\n',total);
fprintf('Correct predictions   : %d\n',correct);
fprintf('Incorrect predictions : %d\n',total-correct);
fprintf('Test accuracy         : %.2f%%\n',accuracy);
fprintf('====================================================\n');

figure('Name','2-D CNN Confusion Matrix','NumberTitle','off','Color','w');
cm=confusionchart(trueLabels,predictedLabels);
cm.Title=sprintf('Four-Class 2-D CNN - Test Accuracy %.2f%%',accuracy);
end
