function fcn_EEGpic(SEG,Ax_eeg,str)
    Fs=200;w=10;zScale=1/150;  
    tc=size(SEG,2)/2;
    eeg=SEG(1:18,tc-w*Fs/2+1:tc+w*Fs/2);
    ekg=-SEG(19,tc-w*Fs/2+1:tc+w*Fs/2);

    tto=tc*Fs-(w/2)*Fs+1;tt1=tc*Fs+(w/2)*Fs;
    tt=tto:tt1;
        
    gap=NaN(1,size(eeg,2));
    seg=eeg;
    seg_disp=[seg(1:4,:);gap;seg(5:8,:);gap;seg(9:12,:);gap;seg(13:16,:);gap;seg(17:18,:);gap;ekg];
   
    M=size(seg_disp,1);
    DCoff=repmat(flipud((1:M)'),1,size(seg_disp,2));
    seg_disp(seg_disp>300)=300;seg_disp(seg_disp<-300)=-300;

    set(0,'CurrentFigure',gcf);set(gcf,'CurrentAxes',Ax_eeg);cla(Ax_eeg)
    hold(Ax_eeg,'on')
        title(str)
        for iSec=1:11
            ta=tto+Fs*(iSec-1);
            line(Ax_eeg,[ta ta], [0 M+1],'linestyle','--','color',[.5 .5 .5])
        end

        plot(Ax_eeg,tt,zScale*seg_disp(1:end-1,:)+DCoff(1:end-1,:),'k','linewidth',1);
        
        ekg_=seg_disp(end,:);ekg_=(ekg_-mean(ekg_))/(eps+std(ekg_));
        plot(Ax_eeg,tt,.2*ekg_+DCoff(end,:),'r','linewidth',1);
        axis off
        set(Ax_eeg,'ylim',[0 M+1],'xlim',[tto tt1+1])
    hold(Ax_eeg,'off')
end
   