function [EEG_clean,isBadChannels]=fcn_cleaningPipeline(EEG,EKG) 
    tmp=load('LOC_18channels.mat');
    Vxy=cell2mat(tmp.LUT(:,2)); 

    tmp=load('BC_LUT_v2.mat');
    BC_LUT=tmp.LUT;
    K=10;Fs=200;win=10*Fs;
    [M,N]=size(EEG);nWin=ceil(N/win);   
    EEG_tmp=zeros(M,nWin*win);EKG_tmp=zeros(1,nWin*win);   
    EEG_tmp(:,1:N)=EEG;EKG_tmp(:,1:N)=EKG;
    EEG_clean=NaN(M-1,nWin*win);
    isBadChannels=NaN(nWin,M-1);
    for i=1:nWin
        a=(i-1)*win+1;b=i*win;     
        eeg=EEG_tmp(:,a:b);
        ekg=EKG_tmp(:,a:b);   
        if isnan(mean(ekg))||mean(abs(ekg))==0
            ekg=mean(eeg,1);
        end
        eeg=fcn_bipolar(eeg); 
        isBad=fcn_isBadChannel_v2(eeg,ekg,Fs,BC_LUT,K);
        eeg_clean=fcn_fixBadChannel(eeg,isBad,Vxy);
        EEG_clean(:,a:b)=eeg_clean;
        isBadChannels(i,:)=isBad;
    end
    EEG_clean=EEG_clean(:,1:N);
end
