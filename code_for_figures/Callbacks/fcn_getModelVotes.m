function ym=fcn_getModelVotes(M,thresh)
    for i=1:size(M,1)
       m=M(i,:);[m1,j1]=max(m);ym(i,1)=j1; 
       m(j1)=0;[m2,j2]=max(m); 
       if m2>(m1-thresh(j1,j2))
           ym(i,1)=j2; 
       else
           ym(i,1)=j1; 
       end
    end
    ym=ym-1;
end
