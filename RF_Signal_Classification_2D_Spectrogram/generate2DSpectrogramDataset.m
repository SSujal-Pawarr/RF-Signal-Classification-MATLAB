function generate2DSpectrogramDataset()
%% Generate simulated complex I/Q signals and 2-D spectrogram images

projectPath = fileparts(mfilename('fullpath'));
outDir = fullfile(projectPath,'data','spectrogram');
if ~isfolder(outDir), mkdir(outDir); end

rng(42);
classes = ["WLAN","BluetoothLE","LoRa","FM"];
N = 1024;
fs = 1e6;

numTrain = 80;
numVal = 20;
numTest = 20;

win = 128;
noverlap = 96;
nfft = 128;

z = simulateRFSignal("WLAN",N,fs);
[S,~,~] = spectrogram(z,win,noverlap,nfft,fs,'centered');
img = single(log1p(abs(S)));
H = size(img,1); W = size(img,2);

XTrain = zeros(H,W,1,numel(classes)*numTrain,'single');
XVal   = zeros(H,W,1,numel(classes)*numVal,'single');
XTest  = zeros(H,W,1,numel(classes)*numTest,'single');

YTrain = categorical(repelem(classes,numTrain)',classes);
YVal   = categorical(repelem(classes,numVal)',classes);
YTest  = categorical(repelem(classes,numTest)',classes);

p = 1;
for c = 1:numel(classes)
    for k = 1:numTrain
        z = simulateRFSignal(classes(c),N,fs);
        XTrain(:,:,1,p) = makeImage(z,fs,win,noverlap,nfft); p=p+1;
    end
end

p = 1;
for c = 1:numel(classes)
    for k = 1:numVal
        z = simulateRFSignal(classes(c),N,fs);
        XVal(:,:,1,p) = makeImage(z,fs,win,noverlap,nfft); p=p+1;
    end
end

p = 1;
for c = 1:numel(classes)
    for k = 1:numTest
        z = simulateRFSignal(classes(c),N,fs);
        XTest(:,:,1,p) = makeImage(z,fs,win,noverlap,nfft); p=p+1;
    end
end

mu = mean(XTrain(:));
sigma = std(XTrain(:));
if sigma < eps, sigma = 1; end

XTrain=(XTrain-mu)/sigma;
XVal=(XVal-mu)/sigma;
XTest=(XTest-mu)/sigma;

dataset = struct();
dataset.XTrain=XTrain; dataset.YTrain=YTrain;
dataset.XValidation=XVal; dataset.YValidation=YVal;
dataset.XTest=XTest; dataset.YTest=YTest;
dataset.classNames=classes;
dataset.N=N; dataset.fs=fs;
dataset.spectrogramWindow=win;
dataset.spectrogramOverlap=noverlap;
dataset.nfft=nfft;
dataset.normalizationMean=mu;
dataset.normalizationStd=sigma;

save(fullfile(outDir,'spectrogramWirelessDataset.mat'),'dataset','-v7.3');

fprintf('Saved dataset.\n');
fprintf('Spectrogram size: %d x %d x 1\n',H,W);
fprintf('Training images: %d\n',numel(YTrain));
fprintf('Validation images: %d\n',numel(YVal));
fprintf('Test images: %d\n',numel(YTest));
end

function img=makeImage(z,fs,win,noverlap,nfft)
[S,~,~]=spectrogram(z,win,noverlap,nfft,fs,'centered');
img=single(log1p(abs(S)));
end

function z=simulateRFSignal(label,N,fs)
t=(0:N-1)'/fs;

switch string(label)
    case "WLAN"
        z=zeros(N,1);
        for k=1:12
            f=(k-6.5)*fs/32;
            sym=randi([0 3],1,ceil(N/64));
            sym=exp(1j*(pi/2)*sym);
            a=repelem(sym,64).'; a=a(1:N);
            z=z+a.*exp(1j*2*pi*f*t);
        end
        z=z+0.25*(randn(N,1)+1j*randn(N,1));

    case "BluetoothLE"
        bits=randi([0 1],ceil(N/32),1);
        nr=repelem(2*bits-1,32); nr=nr(1:N);
        phase=2*pi*cumsum(0.10*fs*nr)/fs;
        z=exp(1j*phase)+0.18*(randn(N,1)+1j*randn(N,1));

    case "LoRa"
        bw=0.30*fs;
        chirpRate=bw/(N/fs);
        phase=2*pi*((-bw/2)*t+0.5*chirpRate*t.^2);
        z=exp(1j*phase)+0.16*(randn(N,1)+1j*randn(N,1));

    case "FM"
        msg=sin(2*pi*0.015*fs*t)+0.35*sin(2*pi*0.00675*fs*t);
        z=exp(1j*1.8*msg)+0.20*(randn(N,1)+1j*randn(N,1));

    otherwise
        error('Unknown class: %s',label);
end

z=z./max(abs(z)+eps);
end
