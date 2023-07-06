function [M, Mk]=fcn_get_mIRR_human_loo(Y)
    pp=[1:5,0];M=zeros(6,6);ct=0;
    for k=1:size(Y, 2)
        yk=Y(:,k);
        Y_=Y;Y_(:, k)=[];
        yref=mode(Y_,2);
        mk=fcn_getCM(yref,yk,pp);
        if sum(isnan(mk(:)))==0
          M=M+mk;Mk{k}=mk;ct=ct+1;
        end
    end
    M=M/ct;
end
