%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 5: Examples of Smooth Pattern Transition for GPD (A) and LRDA (B)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;clear;

addpath('./Callbacks/');
dataDir='./Data/Figure4to6/samples/';
tmp=load('./Data/Figure3/samples.mat');
LUT=tmp.LUT;
Fs=200;[B1,A1]=butter(3,[.5,40]/Fs);[B2,A2]=butter(3,[55,65]/Fs,'stop');

%% set figure
f=figure('units','normalized','position',[0.1000,0.0380,0.4667,0.9333],'MenuBar','none','ToolBar','none','color','w');
ax={subplot('position',[.05,.74,.43,.20]);subplot('position',[.05,.50,.43,.20]);subplot('position',[.05,.26,.43,.20]);subplot('position',[.05,.02,.43,.20]);subplot('position',[.53,.74,.43,.20]);subplot('position',[.53,.50,.43,.20]);subplot('position',[.53,.26,.43,.20]);subplot('position',[.53,.02,.43,.20])};
 
% A. GPD
lut=LUT{3};
for i=1:size(lut,1)
    fileName=lut{i,1};tmp=load([dataDir,fileName]);
    seg=tmp.seg;seg=filtfilt(B1,A1,seg')';seg=filtfilt(B2,A2,seg')';
    eeg=seg(1:19,(20*Fs+1):30*Fs);ekg=seg(20,(20*Fs+1):30*Fs);
    if isnan(var(ekg))||var(ekg)==0
        ekg=mean(eeg,1);
    end
    eeg_clean=fcn_cleaningPipeline(eeg,ekg);
    SEG=[eeg_clean;ekg]; 
    yp=num2str(round(100*lut{i,3}(4))/100);
    if length(yp)==1
        yp=[yp,'.00'];
    else
        yp=[yp,repmat('0',1,4-length(yp))];
    end  
    fcn_EEGpic(SEG,ax{i},['GPD_',num2str(i),'    ',yp]) 
end

hold(ax{1},'on');
    text(ax{1},198900,28,'A','fontsize',25)
hold(ax{1},'off');

%% B. LRDA
lut=LUT{4};
for i=1:size(lut,1)
    fileName=lut{i,1};tmp=load([dataDir,fileName]);
    seg=tmp.seg;
    seg=filtfilt(B1,A1,seg')';seg=filtfilt(B2,A2,seg')';
    eeg=seg(1:19,(20*Fs+1):30*Fs);ekg=seg(20,(20*Fs+1):30*Fs);
    if isnan(var(ekg))||var(ekg)==0
        ekg=mean(eeg,1);
    end
    eeg_clean=fcn_cleaningPipeline(eeg,ekg);
    SEG=[eeg_clean;ekg];
    yp=num2str(round(100*lut{i,3}(5))/100);
    if length(yp)==1
        yp=[yp,'.00'];
    else
        yp=[yp,repmat('0',1,4-length(yp))];
    end
    fcn_EEGpic(SEG,ax{4+i},['LRDA_',num2str(i),'    ',yp])
end
hold(ax{5},'on');
    text(ax{5},198900,28,'B','fontsize',25)
hold(ax{5},'off');

%%
print(gcf,'-r300','-dpng','./Fig5.png');
