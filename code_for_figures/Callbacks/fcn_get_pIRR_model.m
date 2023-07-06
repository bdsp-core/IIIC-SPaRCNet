function [P,PairwiseComparisons,nij,Pij,C,med_mn_mx]=fcn_get_pIRR_model(Y,y_model)
    pp=[1:5,0];pairs=0;P=zeros(6,6);nij=0;expertsUsed=[];
    PairwiseComparisons=zeros(size(Y,2)*2,1);
    for i=1:size(Y,2)
        yi=Y(:,i);
        [pij,c]=fcn_getCM(yi,y_model,pp);
        mn=min(c(:));
        n=sum(~isnan(yi)&~isnan(y_model));
        if sum(isnan(pij(:)))==0&&mn>=10
            pairs=pairs + 1;
            P=P + pij;
            Pij{pairs}=pij;
            nij(pairs,1)=n;
            C{pairs}=c;
            PairwiseComparisons(i)=n;
            expertsUsed=[expertsUsed;i;-1];
        end
        
        [pij,c]=fcn_getCM(y_model,yi,pp);
        mn=min(c(:));
        if sum(isnan(pij(:)))==0&&mn>=10 
            pairs=pairs + 1;
            P=P + pij;
            Pij{pairs}=pij;
            nij(pairs,1)=n;
            C{pairs}=c;
            PairwiseComparisons(i)=n;
            expertsUsed=[expertsUsed;i;-1];
        end
    end
    P=P/pairs;

    haveMoreThan100=zeros(size(C,2),1);
    forMedian=[];
    for i=1:size(C,2)
       c=C{i};
       mn(i)=min(min(c),[],2);
       if mn(i)>100
           haveMoreThan100(i)=1;
           forMedian=[forMedian c(:)];
       end
    end
    med_mn_mx=[median(forMedian(:)),min(forMedian(:)),max(forMedian(:))];
end
