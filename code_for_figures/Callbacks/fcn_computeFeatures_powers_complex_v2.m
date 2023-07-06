function features=fcn_computeFeatures_powers_complex_v2(seg,ekg,Fs)  

    set0=[min(seg,[],2),max(seg,[],2),mean(seg,2),std(seg,[],2),median(seg,2),(prctile(seg,75,2)-prctile(seg,25,2)),(prctile(seg,97.5,2)-prctile(seg,2.5,2)),prctile(seg,2.5,2),prctile(seg,25,2),prctile(seg,75,2),prctile(seg,97.5,2)];
    
    P1=bandpower(seg',Fs,[.5,4])';
    P2=bandpower(seg',Fs,[4,7])';
    P3=bandpower(seg',Fs,[8,15])';
    P4=bandpower(seg',Fs,[16,31])';
    P5=bandpower(seg',Fs,[32,64])';
    P=(P1+P2+P3+P4+P5);

    set1=[pow2db(P1+eps),pow2db(P2+eps),pow2db(P3+eps),pow2db(P4+eps),pow2db(P5+eps),pow2db(P+eps),...
          P1./(P+eps),P2./(P+eps),P3./(P+eps),P4./(P+eps),P5./(P+eps),...
          P1./(P2+eps),P1./(P3+eps),P1./(P4+eps),P1./(P5+eps)...
          P2./(P1+eps),P2./(P3+eps),P2./(P4+eps),P2./(P5+eps),...
          P3./(P1+eps),P3./(P2+eps),P3./(P4+eps),P3./(P5+eps),...
          P4./(P1+eps),P4./(P2+eps),P4./(P3+eps),P4./(P5+eps),...
          P5./(P1+eps),P5./(P2+eps),P5./(P3+eps),P5./(P4+eps)];
        
    entr=NaN(size(seg,1),1);ZCC=NaN(size(seg,1),1);LL=ZCC;
    for i=1:size(seg,1)
        x=seg(i,:);
        entr(i)=entropy(x);
        ZCC(i)=length(fcn_getZeroCrossings(x));
        LL(i) =nanmean(abs(diff(x)));
    end
    set2=[entr,ZCC,LL];
    
    seg_n=(seg-repmat(nanmean(seg,2),1,size(seg,2)))./repmat(nanstd(seg,[],2),1,size(seg,2)+eps);
    entr_n =NaN(size(seg,1),1);
    ZCC_n=NaN(size(seg,1),1);
    LL_n=NaN(size(seg,1),1);
    for i=1:size(seg,1)
        x=seg_n(i,:);
        entr_n(i)=entropy(x);
        ZCC_n(i)=length(fcn_getZeroCrossings(x));
        LL_n(i) =nanmean(abs(diff(x)));
    end
    set2=[set2,entr_n,ZCC_n,LL_n];
    
    Xcorr=NaN(size(seg,1));
    for i=1:(size(seg,1)-1)
        x1=seg(i,:);      
        for j=(i+1):size(seg,1)
           x2=seg(j,:); 
           Xcorr(i,j)=max(abs(xcorr(x1,x2,'coeff')));Xcorr(j,i)=Xcorr(i,j);
        end
    end
    
    set3=NaN(size(seg,1),3);
    for i=1:size(Xcorr,1)
        x=Xcorr(i,:);      
        [x1,~]=sort(x(~isnan(x)),'descend');
        [x2,~]=sort(x(~isnan(x)),'ascend');
        set3(i,1)=mean(x1(1:min(length(x1),3)));
        set3(i,2)=mean(x2(1:min(length(x2),3)));
        set3(i,3)=max(abs(xcorr(seg(i,:),ekg,'coeff')));
    end
    set3(isnan(set3))=0;
    
    features=[set0,set1,set2,set3];
end