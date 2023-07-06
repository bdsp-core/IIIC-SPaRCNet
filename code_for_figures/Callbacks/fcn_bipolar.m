function dataBipolar=fcn_bipolar(data)
    dataBipolar(1,:)=data(1,:)-data(5,:);  % Fp1-F7
    dataBipolar(2,:)=data(5,:)-data(6,:);  % F7-T3
    dataBipolar(3,:)=data(6,:)-data(7,:);  % T3-T5
    dataBipolar(4,:)=data(7,:)-data(8,:);  % T5-O1

    dataBipolar(5,:)=data(12,:)-data(16,:);  % Fp2-F8
    dataBipolar(6,:)=data(16,:)-data(17,:);  % F8-T4
    dataBipolar(7,:)=data(17,:)-data(18,:);  % T4-T6
    dataBipolar(8,:)=data(18,:)-data(19,:);  % T6-O2

    dataBipolar(9,:)=data(1,:)-data(2,:);   % Fp1-F3
    dataBipolar(10,:)=data(2,:)-data(3,:);  % F3-C3
    dataBipolar(11,:)=data(3,:)-data(4,:);  % C3-P3
    dataBipolar(12,:)=data(4,:)-data(8,:);  % P3-O1

    dataBipolar(13,:)=data(12,:)-data(13,:);  % Fp2-F4
    dataBipolar(14,:)=data(13,:)-data(14,:);  % F4-C4
    dataBipolar(15,:)=data(14,:)-data(15,:);  % C4-P4
    dataBipolar(16,:)=data(15,:)-data(19,:);  % P4-O2

    dataBipolar(17,:)=data(9,:)-data(10,:);   % Fz-Cz
    dataBipolar(18,:)=data(10,:)-data(11,:);  % Cz-Pz
end