function zcc=fcn_getZeroCrossings(x)
    zcc=find((x(1:end-1)<=0&x(2:end)>0)|(x(1:end-1)>=0&x(2:end)<0));
end