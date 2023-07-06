function eeg_clean=fcn_fixBadChannel(eeg,badScr,LOC)
    % eeg: 18 channel bipolar 
    % badScr: [0] good [1] bad 
    
    [M,N]=size(eeg);
    idx_good=find(badScr==0);
    if isempty(idx_good)
        eeg_clean=zeros(size(eeg));
    else
        idx_bad=find(badScr==1);
        eeg_clean= eeg;
        eeg_clean(idx_bad,:)=NaN;
        while ~isempty(idx_bad)  
            x=LOC(idx_good,1);y=LOC(idx_good,2);
            xq=LOC(idx_bad,1);yq=LOC(idx_bad,2);           
            vq_1=griddata(x,y,eeg_clean(idx_good,1),xq,yq);              
            if ~isempty(vq_1) 
                for jj=1:N 
                    v=eeg_clean(idx_good,jj);
                    eeg_clean(idx_bad,jj)=griddata(x,y,v,xq,yq); 
                end
            end            
            idx_bad_old=idx_bad;
            badScr=isnan(eeg_clean(:,1));           
            idx_good=find(badScr==0);
            idx_bad=find(badScr==1);  
            if isempty(idx_bad)
                break
            else
                if length(idx_bad)==length(idx_bad_old)                
                    avg_ll=nanmean(eeg_clean(1:4,:),1);
                    avg_rl=nanmean(eeg_clean(5:8,:),1);
                    avg_lp=nanmean(eeg_clean(9:12,:),1);
                    avg_rp=nanmean(eeg_clean(13:16,:),1);
                    avg_cc=nanmean(eeg_clean(17:18,:),1);                   
                    for ii=1:M
                        if badScr(ii)==1
                            switch ii
                                case {1,2,3,4}
                                    eeg_ii=avg_ll;
                                case {5,6,7,8}
                                    eeg_ii=avg_rl;
                                case {9,10,11,12}
                                    eeg_ii=avg_lp;
                                case {13,14,15,16}
                                    eeg_ii=avg_rp; 
                                case {17,18}
                                    eeg_ii=avg_cc; 
                            end                        
                            if isnan(mean(eeg_ii))
                                eeg_ii=zeros(1,N);
                            end
                            eeg_clean(ii,:)=eeg_ii;
                        end
                    end
                    break
                end
            end
        end
    end
end
