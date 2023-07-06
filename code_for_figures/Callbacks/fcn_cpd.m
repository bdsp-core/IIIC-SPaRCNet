function [icp_,P_,flags1,flags2]=fcn_cpd(sdata,thr_cp)
    nn=size(sdata{1},2);
    P=mean(pow2db(cell2mat(sdata(:,1))+eps),1);
    P_=smooth(P,10,'sgolay')';
    P_(P_>25)=25;P_(P_<-15)=-15;

    [icp,~]=findchangepts(P_,'Statistic','mean','MinThreshold',thr_cp*var(P_));
    flags1=zeros(nn,1);flags1(icp)=1;flags1(1)=1;flags1(end)=-1;

    icp_=unique([icp,1,nn]); 
    icp_center=floor((icp_(1:end-1)+icp_(2:end))/2);
    flags2=zeros(nn,1);flags2(icp_center)=1;
end