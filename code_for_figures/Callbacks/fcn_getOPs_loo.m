function [SEN,FPR,PPV]=fcn_getOPs_loo(Y,p)
    SEN=NaN(length(p),size(Y,2));FPR=NaN(length(p),size(Y,2));PPV=NaN(length(p),size(Y,2));
    for i=1:length(p)
        for j=1:size(Y,2)
            idx_others=setdiff(1:size(Y,2),j);
            Y_j=Y(:,idx_others);
            yy=mode(Y_j,2);
            yt_full=(yy==p(i));

            temp=Y(:,j);ind=find(~isnan(temp));temp=temp(ind);       
            yi=(temp==p(i));
            yt=yt_full(ind);     

            SEN(i,j)=100*sum(yi==1&yt==1)/sum(yt==1); 
            FPR(i,j)=100*sum(yi==1&yt==0)/sum(yt==0); 
            PPV(i,j)=100*sum(yi==1&yt==1)/sum(yi==1); 
        end
    end
    PPV(isnan(PPV))=1;SEN(isnan(SEN))=1;FPR(isnan(FPR))=0;
end
