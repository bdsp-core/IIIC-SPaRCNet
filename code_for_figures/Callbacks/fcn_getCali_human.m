function cali_idx=fcn_getCali_human(Y,p,K,M)
    b_den=sum(~isnan(Y),2); 
    Yr=Y(b_den>=10,:); 
    Ne=size(Yr,2); 
    bin_edges=(0:K)/K;n=NaN(Ne,K);
    for i=1:Ne      
        yi=Yr(:,i); 
        Yr_=Yr(:,setdiff(1:Ne,i));
        b_num=sum(Yr_==p,2);
        b_den=sum(~isnan(Yr_),2); 
        b=b_num./b_den; 

        for j=1:K
           bin_left=bin_edges(j);bin_right= bin_edges(j+1);
           bin_center(j)=(bin_left+bin_right)/2; 
           ind=find(b>bin_left&b<=bin_right&~isnan(yi));
           n(i,j)=100*sum(yi(ind)==p)/length(ind); 
           if length(ind)<10
               n(i,j)=nan; 
           end
        end
    end

    ns=[];th=linspace(-20,20,1000); 
    for k=1:size(n,1)
        yy=n(k,:);idx1=find(~isnan(yy));
        yy=[0,yy(idx1),100];xx=[0,100*bin_center(idx1),100];        
        best=inf;best_th=0; 
        for j=1:length(th) 
            pr=min(xx/100,(1-0.001)); 
            z=log((pr)./(1-pr))+th(j); 
            yh=100./(1+exp(-z)); 
            C=sum((yy-yh).^2); 
            if C<best 
                best=C;best_th=th(j);                      
            end
        end    
        xxi=linspace(0,100,M);
        pr=xxi/100; 
        z=log((pr)./(1-pr))+best_th;
        yh=100./(1+exp(-z));
        ns(k,:)=yh; 
        THRESH(1,k)=best_th; 
    end
    
    cali_thr=THRESH;
    cali_idx=NaN(size(cali_thr));
    x=linspace(eps,1-eps,1000); 
    for k =1:size(n,1)
        th=THRESH(k);
        z=log(x./(1-x))+th; 
        y=1./(1+exp(-z)); 
        d=y-x; 
        a=trapz(x,d); 
        cali_idx(k)=a/0.5; 
    end
end
