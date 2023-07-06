function [P,PairwiseComparisons,nij,Pij,C,med_mn_mx]=fcn_get_pIRR_human(Y)
    pp=[1:5,0];pairs=0;P=zeros(6,6);nij=0;expertsUsed=[];
    PairwiseComparisons=zeros(size(Y,2),size(Y,2));
    for i=1:size(Y,2)
        yi=Y(:,i);
        for j=1:size(Y,2)
            if i~=j         
                yj=Y(:,j);     
                [pij,c]=fcn_getCM(yi,yj,pp);
                mn=min(min(c),[],2);
                n=sum(~isnan(yi)&~isnan(yj));
                if sum(isnan(pij(:)))==0&&mn>=10
                    pairs=pairs + 1;
                    P=P + pij;
                    Pij{pairs}=pij;
                    nij(pairs,1)=n;
                    C{pairs}=c;
                    PairwiseComparisons(i,j)=n;
                    expertsUsed=[expertsUsed;i;j];
                end
            end                
        end
    end
    P=P/pairs;

    haveMoreThan100=zeros(size(C,2),1);forMedian=[];
    for i=1:size(C,2)
       c=C{i};
       mn(i)=min(min(c),[],2);
       if mn(i)>100
           haveMoreThan100(i)=1;
           forMedian=[forMedian c(:)];
       end
    end
    med_mn_mx=[median(forMedian(:));min(forMedian(:));max(forMedian(:))];
end
