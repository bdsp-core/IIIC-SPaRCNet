function [data,Fs,channels]=fcn_preprocess(data,Fs,channels,filterOn)
    channels_1020={'Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2'};
    idx=[];
    for j=1:length(channels_1020)
        jj=find(ismember(lower(channels),lower(channels_1020{j}))==1);
        if isempty(jj)
            keyboard     
        else
            idx=[idx; jj];
        end  
    end
    eeg=data(idx,:);eeg(isnan(eeg))=9999;

    channels_ekg={'EKG','ECG','EKGL','ECGL','EKGR','ECGR'};
    idx=[];
    for j=1:length(channels_ekg)
        jj=find(ismember(lower(channels),lower(channels_ekg{j}))==1);
        if ~isempty(jj)
            idx=[idx; jj];
        end  
    end

    if isempty(idx)
        ekg=NaN(1,size(data,2));
    else
        if length(idx)==1
            ekg=data(idx,:);
            ekg(isnan(ekg))=0;
        else
            ekg=data(idx,:);
            ekg=ekg(1,:) - ekg(2,:);
            ekg(isnan(ekg))=0;
        end
    end
    data=[eeg;ekg];channels=[channels_1020,{'EKG'}]';
    if round(Fs)~=200
        data=resample(data',200,Fs)';Fs=200;
    end
    
    if filterOn
        [B1,A1]=butter(3,[.5,40]/(.5*Fs));
        [B2,A2]=iirnotch(60/(.5*Fs),60/(.5*Fs*35));
        try
            data=filtfilt(B1,A1,data')';data=filtfilt(B2,A2,data')';
        catch err
            data=filter(B1,A1,data')';data=filter(B2,A2,data')';
        end
    end
end
