function demoWirelessIQ()
clc; close all; root=fileparts(mfilename('fullpath')); addpath(genpath(root));
load(fullfile(root,'data','labeled_iq','labeledWirelessIQDataset.mat'),'dataset');
load(fullfile(root,'models','wirelessIQClassifier.mat'),'net','classNames'); classNames=string(classNames(:));
while true
 fprintf('\n====================================================\n       FOUR-CLASS 1-D RF CLASSIFICATION DEMO\n====================================================\n');
 fprintf('\nAvailable input options:\n\n1. WLAN test sample\n2. BluetoothLE test sample\n3. LoRa test sample\n4. FM test sample\n5. Random held-out test sample\n6. Custom I/Q MAT file\n7. Evaluate complete test set\n8. Exit\n');
 choice=input('\nEnter choice (1-8): ');
 if choice==8, fprintf('Exiting demo.\n'); return; end
 if choice==7, testWirelessClassifier; continue; end
 if choice>=1 && choice<=4
  target=classNames(choice); ix=find(string(dataset.YTest)==target,1); iq=dataset.XTest{ix}.';
  source="Held-out test sample: "+target; known=target;
 elseif choice==5
  ix=randi(numel(dataset.XTest)); iq=dataset.XTest{ix}.'; source="Random held-out test sample"; known=string(dataset.YTest(ix));
 elseif choice==6
  [f,p]=uigetfile('*.mat','Select custom I/Q MAT file'); if isequal(f,0), continue; end
  S=load(fullfile(p,f)); iq=extractIQ(S); if size(iq,2)~=2 && size(iq,1)==2, iq=iq.'; end
  if size(iq,2)~=2, error('MAT file must contain numeric variable iq, IQ, X, or rawIQ with 2 columns [I Q].'); end
  source="Custom MAT file: "+string(f); known="Unknown";
 else, fprintf('Invalid choice.\n'); continue; end
 if size(iq,1)~=dataset.N, error('Input must contain exactly %d samples.',dataset.N); end
 I=double(iq(:,1)); Q=double(iq(:,2));
 fprintf('\n====================================================\n                 INPUT INFORMATION\n====================================================\nInput source : %s\nInput samples: %d\nInput columns: I and Q\n====================================================\n',source,size(iq,1));
 fprintf('\n[1/3] Preparing I/Q sequence for 1-D CNN...\n'); seq={single(iq.')};
 fprintf('[2/3] Running 1-D CNN classification...\n'); [pred,scores]=classify(net,seq); scores=scores(:);
 fprintf('[3/3] Displaying classification results...\n\n====================================================\n                 1-D CNN OUTPUT\n====================================================\nInput source: %s\nPredicted class: %s\n',source,string(pred));
 if known~="Unknown", fprintf('Known label: %s\nPrediction correct: %s\n',known,string(string(pred)==known));
 else, fprintf('Known label: Unknown (custom input has no ground-truth label)\n'); end
 fprintf('Class scores:\n'); for k=1:min(numel(classNames),numel(scores)), fprintf('  %-15s %.4f\n',classNames(k),scores(k)); end
 fprintf('====================================================\n');
 % Figure 1: I and Q waveforms
    fig1 = figure('Name','1-D Input I/Q','NumberTitle','off','Color','w');
    tl = tiledlayout(fig1,2,1,'Padding','compact','TileSpacing','compact');

    ax1 = nexttile(tl);
    plot(ax1,I,'LineWidth',1);
    title(ax1,'Input RF Signal - In-phase (I)','Color','k','FontWeight','bold');
    xlabel(ax1,'Sample Index','Color','k');
    ylabel(ax1,'Amplitude','Color','k');
    grid(ax1,'on'); xlim(ax1,[1 numel(I)]);
    set(ax1,'Color','w','XColor','k','YColor','k', ...
        'GridColor',[0.55 0.55 0.55],'FontWeight','bold');

    ax2 = nexttile(tl);
    plot(ax2,Q,'LineWidth',1);
    title(ax2,'Input RF Signal - Quadrature (Q)','Color','k','FontWeight','bold');
    xlabel(ax2,'Sample Index','Color','k');
    ylabel(ax2,'Amplitude','Color','k');
    grid(ax2,'on'); xlim(ax2,[1 numel(Q)]);
    set(ax2,'Color','w','XColor','k','YColor','k', ...
        'GridColor',[0.55 0.55 0.55],'FontWeight','bold');

    title(tl,'Raw I/Q Input Signal','Color','k','FontWeight','bold');

    % Figure 2: I/Q sample distribution
    fig2 = figure('Name','1-D I/Q Distribution','NumberTitle','off','Color','w');
    ax3 = axes(fig2);
    plot(ax3,I,Q,'.','MarkerSize',7);
    title(ax3,'I-Q Sample Distribution','Color','k','FontWeight','bold');
    xlabel(ax3,'In-phase (I)','Color','k');
    ylabel(ax3,'Quadrature (Q)','Color','k');
    grid(ax3,'on'); axis(ax3,'equal');
    set(ax3,'Color','w','XColor','k','YColor','k', ...
        'GridColor',[0.55 0.55 0.55],'FontWeight','bold');

    % Figure 3: 1-D frequency spectrum (FFT magnitude versus frequency)
    fig3 = figure('Name','1-D Input Spectrum','NumberTitle','off','Color','w');
    z = complex(I,Q);
    nf = max(2048,2^nextpow2(numel(z)));
    Z = fftshift(fft(z,nf));
    f = (-nf/2:nf/2-1)*(dataset.fs/nf);
    magDB = 20*log10(abs(Z)+1e-8);
    ax4 = axes(fig3);
    plot(ax4,f/1e3,magDB,'LineWidth',1);
    title(ax4,'Input Signal Frequency Spectrum','Color','k','FontWeight','bold');
    xlabel(ax4,'Frequency (kHz)','Color','k');
    ylabel(ax4,'Magnitude (dB)','Color','k');
    grid(ax4,'on');
    set(ax4,'Color','w','XColor','k','YColor','k', ...
        'GridColor',[0.55 0.55 0.55],'FontWeight','bold');

    % Figure 4: classification scores
    fig4 = figure('Name','1-D CNN Classification Scores','NumberTitle','off','Color','w');
    ax5 = axes(fig4);
    bar(ax5,scores);
    xticks(ax5,1:numel(scores));
    xticklabels(ax5,classNames);
    xlabel(ax5,'RF Signal Class','Color','k');
    ylabel(ax5,'Model Score','Color','k');
    title(ax5,'1-D CNN Classification Scores','Color','k','FontWeight','bold');
    ylim(ax5,[0 1.05]);
    grid(ax5,'on');
    set(ax5,'Color','w','XColor','k','YColor','k', ...
        'GridColor',[0.55 0.55 0.55],'FontWeight','bold');
 fprintf('\n====================================================\n             DEMONSTRATION COMPLETED\n====================================================\nInput          : Raw I/Q signal samples\nRepresentation: 1-D I/Q sequence\nAI model       : Trained 1-D CNN\nPrediction     : %s\n\nProcessing flow:\nRaw I/Q -> 1-D sequence -> 1-D CNN -> Predicted RF class\n\nNote: Results are based on simulated I/Q data.\nThey do not imply real-world RF performance.\n====================================================\n',string(pred));
end
end
function iq=extractIQ(S)
for n=["iq","IQ","X","rawIQ"]
 if isfield(S,n) && isnumeric(S.(n)) && ismatrix(S.(n)), iq=double(S.(n)); return; end
end
error('MAT file must contain numeric variable iq, IQ, X, or rawIQ.');
end
