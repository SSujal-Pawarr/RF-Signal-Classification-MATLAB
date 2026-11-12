%% Create a custom simulated I/Q MAT file
clc; clear;

projectPath=fileparts(mfilename('fullpath'));
outFile=fullfile(projectPath,'data','labeled_iq','custom_test_input.mat');

N=1024; fs=1e6;
t=(0:N-1)'/fs;
msg=sin(2*pi*0.015*fs*t)+0.35*sin(2*pi*0.00675*fs*t);
z=exp(1j*1.8*msg)+0.20*(randn(N,1)+1j*randn(N,1));
z=z./max(abs(z)+eps);

iq=[real(z),imag(z)];
save(outFile,'iq','-v7');

fprintf('Custom I/Q file saved:\n%s\n',outFile);
fprintf('Size: %d x %d\n',size(iq,1),size(iq,2));
