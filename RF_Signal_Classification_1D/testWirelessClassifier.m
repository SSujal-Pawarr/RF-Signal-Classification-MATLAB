function testWirelessClassifier()
root=fileparts(mfilename('fullpath')); load(fullfile(root,'data','labeled_iq','labeledWirelessIQDataset.mat'),'dataset');
load(fullfile(root,'models','wirelessIQClassifier.mat'),'net');
p=classify(net,dataset.XTest); y=dataset.YTest; acc=100*sum(p==y)/numel(y);
fprintf('\n====================================================\n                 1-D TEST RESULTS\n====================================================\n');
fprintf('Total test signals    : %d\nCorrect predictions   : %d\nIncorrect predictions : %d\nTest accuracy         : %.2f%%\n',numel(y),sum(p==y),sum(p~=y),acc);
fprintf('====================================================\n');
figure('Name','1-D CNN Confusion Matrix','Color','w'); cm=confusionchart(y,p); cm.FontColor='black'; cm.Title=sprintf('Four-Class 1-D CNN - Test Accuracy %.2f%%',acc);
end
