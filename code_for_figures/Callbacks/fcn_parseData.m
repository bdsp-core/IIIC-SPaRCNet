function SEG=fcn_parseData(dataSize,seg,Fs,t_left,t_right,win_time)
    M=dataSize(1);
    N=dataSize(2);
    SEG=zeros(M,win_time*Fs);
    if t_left<1&&t_right<N
        a=1-t_left+1;b=size(SEG,2);
        SEG(:,a:b)=seg;
    elseif t_left<1&&t_right>=N
        a=1-t_left+1;b=a+N-1;
        SEG(:,a:b)=seg;
    elseif t_left>=1&&t_right<N
        SEG=seg;
    elseif t_left>=1&&t_right>=N
        a=1;b=N-t_left+1;
        SEG(:,a:b)=seg;
    else
        keyboard
    end
end
