function fcn_plotEEGseg(f,Ax_EEG,eegFullPath,idx_event,montage,labelshow,str_event)
    ww=20;Fs=200;
    matObj=matfile(eegFullPath);
    matObj.Properties.Writable=true;
    size_data=size(matObj,'data');
    N=size_data(2);
    tmp=load(eegFullPath,'Fs','channels');
    channels=tmp.channels';

    t_center=idx_event*2;t_left =(t_center*Fs)-(ww/2)*Fs+1;t_right=(t_center*Fs)+(ww/2)*Fs;
    idx1=max(t_left,1);idx2=min(t_right,N);

    seg_raw =matObj.data(:,idx1:idx2);
    seg=fcn_preprocess(seg_raw,Fs,channels,1);

    SEG_20sec=fcn_parseData([size(seg,1),N],seg,Fs,t_left,t_right,ww);
    seg=SEG_20sec(:,5*Fs+1:15*Fs);eeg=seg(1:19,:);ekg=seg(20,:);
    
    w=10;
    zScale=1/150;
    channel_withspace_bipolar ={'Fp1-F7','F7-T3','T3-T5','T5-O1','','Fp2-F8','F8-T4','T4-T6','T6-O2','','Fp1-F3','F3-C3','C3-P3','P3-O1','','Fp2-F4','F4-C4','C4-P4','P4-O2','','Fz-Cz'  'Cz-Pz','','EKG'};
    tto=1;tt1=w*Fs;
    tt=tto:tt1;
    gap=NaN(1,size(eeg,2));
    switch montage
        case 'L-Bipolar'
            seg=fcn_bipolar(eeg);
            seg_disp=[seg(1:4,:);gap;seg(5:8,:);gap;seg(9:12,:);gap;seg(13:16,:);gap;seg(17:18,:);gap;ekg];
            channel_withspace=channel_withspace_bipolar;

        case 'Average' 
            seg=eeg-repmat(mean(eeg,1),size(eeg,1),1);
            seg_disp=[seg(1:8,:);gap;seg(9:11,:);gap;seg(12:19,:);gap;ekg];
            channel_withspace=channel_withspace_average;

        case 'Monopolar'
            seg= eeg;
            seg_disp=[seg(1:8,:);gap;seg(9:11,:);gap;seg(12:19,:);gap;ekg];
            channel_withspace=channel_withspace_monopolar;
    end
    M=size(seg_disp,1);DCoff=repmat(flipud((1:M)'),1,size(seg_disp,2));

    set(f,'CurrentAxes',Ax_EEG);cla(Ax_EEG)
    hold(Ax_EEG,'on')
        for iSec=1:round((tt1-tto+1)/Fs)+1
            ta=tto+Fs*(iSec-1);
            line([ta ta], [0 M+1],'linestyle','--','color',[.5 .5 .5])
        end

        plot(Ax_EEG,tt,zScale*seg_disp(1:end-1,:)+DCoff(1:end-1,:),'k');
        ekg_=seg_disp(end,:);ekg_=(ekg_-mean(ekg_))/(eps+std(ekg_));
        plot(Ax_EEG,tt,.2*ekg_+DCoff(end,:),'r');
        set(Ax_EEG,'box','off','ylim',[0 M+1],'xlim',[tto tt1+1],'xtick',round(tt(1):2*Fs:tt(end)),'xticklabel',[])

        if labelshow
            for iCh=1:length(channel_withspace)
                ta=DCoff(iCh);
                text(Ax_EEG,tt(1)-Fs/20,ta,channel_withspace(iCh),'fontsize',7,'HorizontalAlignment','right','VerticalAlignment','middle')
            end
        end      
        text(Ax_EEG,tt(1),0,[str_event,'-00:00:05'],'fontsize',10,'HorizontalAlignment','left','VerticalAlignment','top')
        text(Ax_EEG,(tt(1)+tt(end))/2,0,str_event,'fontsize',10,'HorizontalAlignment','left','VerticalAlignment','top')
        text(Ax_EEG,tt(end),0,[str_event,'+00:00:05'],'fontsize',10,'HorizontalAlignment','right','VerticalAlignment','top')
        axis off
    hold(Ax_EEG,'off')
end
