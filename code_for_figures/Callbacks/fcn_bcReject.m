function bc=fcn_bcReject(features,LUT_BG,K)
    bc=zeros(1,18);
    for ch=1:18
        for k=1:K
            idx=LUT_BG{k,1};
            thr=LUT_BG{k,5};
            rho=LUT_BG{k,3};
            x=features(ch,idx)*sign(rho);
            if x<thr
                bc(ch)=1;
                break
            end
        end
    end
end
