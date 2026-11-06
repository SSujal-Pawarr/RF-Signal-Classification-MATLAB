function runProject()
clc; root=fileparts(mfilename('fullpath')); addpath(genpath(root));
dataFile=fullfile(root,'data','labeled_iq','labeledWirelessIQDataset.mat');
modelFile=fullfile(root,'models','wirelessIQClassifier.mat');
fprintf('\n====================================================\n');
fprintf('       1-D RF SIGNAL CLASSIFICATION PROJECT\n');
fprintf('====================================================\n');
if ~isfile(dataFile)
 fprintf('\n[1/3] Generating simulated four-class I/Q dataset...\n'); generateLabeledWirelessDataset;
else, fprintf('\n[1/3] Reusing existing dataset.\n'); end
if ~isfile(modelFile)
 fprintf('\n[2/3] Training 1-D CNN...\n'); trainWirelessClassifier;
else, fprintf('\n[2/3] Reusing existing trained 1-D CNN.\n'); end
fprintf('\n[3/3] Opening classification demo...\n'); demoWirelessIQ;
end
