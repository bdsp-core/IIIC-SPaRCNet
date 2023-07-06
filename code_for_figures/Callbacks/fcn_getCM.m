function [CM,c]=fcn_getCM(yref,yquery,pp)
    CM=zeros(6,6);c=CM;
    for i=1:6
        patternA=pp(i);
        for j=1:6
            patternB=pp(j); 
            n=sum(yref==patternA&yquery==patternB); 
            d=sum(yref==patternA&~isnan(yquery)&~isnan(yref));  
            CM(i,j)=n/d;c(i,j)=d; 
        end
    end
end