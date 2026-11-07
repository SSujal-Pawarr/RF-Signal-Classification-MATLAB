function generateLabeledWirelessDataset()
clc; rng(20261019);
root=fileparts(mfilename('fullpath'));
outFile=fullfile(root,'data','labeled_iq','labeledWirelessIQDataset.mat');
if ~isfolder(fileparts(outFile)), mkdir(fileparts(outFile)); end
N=1024; fs=1e6; labels=["WLAN","BluetoothLE","LoRa","FM"]; nPer=240;
X=cell(numel(labels)*nPer,1); Y=strings(numel(labels)*nPer,1); q=1;
for c=1:numel(labels)
 fprintf('Generating %-12s samples...\n',labels(c));
 for n=1:nPer
  z=simulateRFSignal(labels(c),N,fs);
  X{q}=single([real(z).';imag(z).']); Y(q)=labels(c); q=q+1;
 end
end
p=randperm(numel(Y)); X=X(p); Y=Y(p); n=numel(Y); nt=floor(.70*n); nv=floor(.15*n);
dataset.XTrain=X(1:nt); dataset.YTrain=categorical(Y(1:nt),labels);
dataset.XValidation=X(nt+1:nt+nv); dataset.YValidation=categorical(Y(nt+1:nt+nv),labels);
dataset.XTest=X(nt+nv+1:end); dataset.YTest=categorical(Y(nt+nv+1:end),labels);
dataset.N=N; dataset.fs=fs; dataset.classNames=labels;
save(outFile,'dataset','-v7.3');
fprintf('Saved %s\nTrain: %d | Validation: %d | Test: %d\n',outFile,numel(dataset.YTrain),numel(dataset.YValidation),numel(dataset.YTest));
end
function z=simulateRFSignal(label,N,fs)
t=(0:N-1)'/fs;
switch string(label)
case "WLAN"
 z=zeros(N,1);
 for k=1:12
  f=(k-5.5)*fs/32; s=exp(1j*(pi/2)*randi([0 3],1,16));
  a=repelem(s,64).'; z=z+a(1:N).*exp(1j*2*pi*f*t);
 end
 z=z.*(.75+.20*sin(2*pi*1200*t))+.30*(randn(N,1)+1j*randn(N,1));
case "BluetoothLE"
 b=randi([0 1],32,1); nr=repelem(2*b-1,32); ph=2*pi*cumsum(.10*fs*nr)/fs;
 z=exp(1j*ph).*(.85+.12*sin(2*pi*900*t))+.28*(randn(N,1)+1j*randn(N,1));
case "LoRa"
 bw=.30*fs; rate=bw/(N/fs); ph=2*pi*((-bw/2)*t+.5*rate*t.^2);
 z=exp(1j*ph).*(.90+.08*sin(2*pi*600*t))+.25*(randn(N,1)+1j*randn(N,1));
case "FM"
 msg=sin(2*pi*.015*fs*t)+.35*sin(2*pi*.00675*fs*t);
 z=exp(1j*1.8*msg).*(.82+.16*sin(2*pi*500*t))+.27*(randn(N,1)+1j*randn(N,1));
otherwise, error('Unknown class.');
end
z=z./max(abs(z)+eps);
end
