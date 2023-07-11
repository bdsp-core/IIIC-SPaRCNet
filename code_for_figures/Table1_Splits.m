%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Table 1:  Data splits and human performance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;clear;
 
addpath('./Callbacks/');
pp=[1:5,0];

tmp=load('./Data/Table1/patient_demo.mat');
lut_pat=tmp.res;

%% Table 1 part A - data splits
% Dataset 1: Real+SP-spread
tmp=load('./Data/Table1/dataset1.mat');keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6,1])';p=round(1E3*c/sum(c))/10;
sex=lut_pat(ismember(lut_pat(:,1),pats),2);nFemale=sum(ismember(sex,'Female'));
age=cell2mat(lut_pat(ismember(lut_pat(:,1),pats),3));
str1=['Dataset 1: ',num2str(length(raters)),' raters ',num2str(length(pats)),' patients ',num2str(length(y)),' samples'];str8=['Age    ',num2str(round(10*nanmean(age))/10),' (',num2str(round(10*nanstd(age))/10),')'];
str9=['Sex /F ',num2str(nFemale),' (',num2str(round(1E3*nFemale/length(pats))/10),'%)'];
str2=['SZ     ',num2str(c(1)),' (',num2str(p(1)),'%)'];
str3=['LPD    ',num2str(c(2)),' (',num2str(p(2)),'%)'];
str4=['GPD    ',num2str(c(3)),' (',num2str(p(3)),'%)'];
str5=['LRDA   ',num2str(c(4)),' (',num2str(p(4)),'%)'];
str6=['GRDA   ',num2str(c(5)),' (',num2str(p(5)),'%)'];
str7=['Other  ',num2str(c(6)),' (',num2str(p(6)),'%)'];
disp('-----------------------------------------------------')
disp(str1);disp(str8);disp(str9);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
% Dataset 2: Real+SP-spread+UMAP-spread
tmp=load('./Data/Table1/dataset2.mat');keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6,1])';p=round(1E3*c/sum(c))/10;
sex=lut_pat(ismember(lut_pat(:,1),pats),2);nFemale=sum(ismember(sex,'Female'));
age=cell2mat(lut_pat(ismember(lut_pat(:,1),pats),3));
str1=['Dataset 2: ',num2str(length(raters)),' raters ',num2str(length(pats)),' patients ',num2str(length(y)),' samples'];str8=['Age    ',num2str(round(10*nanmean(age))/10),' (',num2str(round(10*nanstd(age))/10),')'];
str9=['Sex /F ',num2str(nFemale),' (',num2str(round(1E3*nFemale/length(pats))/10),'%)'];
str2=['SZ     ',num2str(c(1)),' (',num2str(p(1)),'%)'];
str3=['LPD    ',num2str(c(2)),' (',num2str(p(2)),'%)'];
str4=['GPD    ',num2str(c(3)),' (',num2str(p(3)),'%)'];
str5=['LRDA   ',num2str(c(4)),' (',num2str(p(4)),'%)'];
str6=['GRDA   ',num2str(c(5)),' (',num2str(p(5)),'%)'];
str7=['Other  ',num2str(c(6)),' (',num2str(p(6)),'%)'];
disp('-----------------------------------------------------')
disp(str1);disp(str8);disp(str9);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
% Dataset 3: Real+SP-spread
tmp=load('./Data/Table1/dataset3.mat');keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6,1])';p=round(1E3*c/sum(c))/10;
sex=lut_pat(ismember(lut_pat(:,1),pats),2);nFemale=sum(ismember(sex,'Female'));
age=cell2mat(lut_pat(ismember(lut_pat(:,1),pats),3));
str1=['Dataset 3: ',num2str(length(raters)),' raters ',num2str(length(pats)),' patients ',num2str(length(y)),' samples'];str8=['Age    ',num2str(round(10*nanmean(age))/10),' (',num2str(round(10*nanstd(age))/10),')'];
str9=['Sex /F ',num2str(nFemale),' (',num2str(round(1E3*nFemale/length(pats))/10),'%)'];
str2=['SZ     ',num2str(c(1)),' (',num2str(p(1)),'%)'];
str3=['LPD    ',num2str(c(2)),' (',num2str(p(2)),'%)'];
str4=['GPD    ',num2str(c(3)),' (',num2str(p(3)),'%)'];
str5=['LRDA   ',num2str(c(4)),' (',num2str(p(4)),'%)'];
str6=['GRDA   ',num2str(c(5)),' (',num2str(p(5)),'%)'];
str7=['Other  ',num2str(c(6)),' (',num2str(p(6)),'%)'];
disp('-----------------------------------------------------')
disp(str1);disp(str8);disp(str9);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
% Dataset 4: Real+SP-spread
tmp=load('./Data/Table1/dataset4.mat');keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6,1])';p=round(1E3*c/sum(c))/10;
sex=lut_pat(ismember(lut_pat(:,1),pats),2);nFemale=sum(ismember(sex,'Female'));
age=cell2mat(lut_pat(ismember(lut_pat(:,1),pats),3));
str1=['Dataset 4: ',num2str(length(raters)),' raters ',num2str(length(pats)),' patients ',num2str(length(y)),' samples'];
str8=['Age    ',num2str(round(10*nanmean(age))/10),' (',num2str(round(10*nanstd(age))/10),')'];
str9=['Sex /F ',num2str(nFemale),' (',num2str(round(1E3*nFemale/length(pats))/10),'%)'];
str2=['SZ     ',num2str(c(1)),' (',num2str(p(1)),'%)'];
str3=['LPD    ',num2str(c(2)),' (',num2str(p(2)),'%)'];
str4=['GPD    ',num2str(c(3)),' (',num2str(p(3)),'%)'];
str5=['LRDA   ',num2str(c(4)),' (',num2str(p(4)),'%)'];
str6=['GRDA   ',num2str(c(5)),' (',num2str(p(5)),'%)'];
str7=['Other  ',num2str(c(6)),' (',num2str(p(6)),'%)'];
disp('-----------------------------------------------------')
disp(str1);disp(str8);disp(str9);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
%% Table 1 part B - human performance
tmp=load('./Data/Table1/dataset3.mat');
Y=tmp.Y;nBins=10;K=100;cali_idx_3=NaN(length(pp),20);
[op_sen_3,op_fpr_3,op_ppv_3]=fcn_getOPs_loo(Y,pp); 
for i=1:length(pp);cali_idx_3(i,:)=fcn_getCali_human(Y,(i-1),nBins,K);end
cali_idx_3=100*cali_idx_3([2:6,1],:);

tmp=load('./Data/Table1/dataset4.mat');
Y=tmp.Y;cali_idx_4=NaN(length(pp),20);
[op_sen_4,op_fpr_4,op_ppv_4]=fcn_getOPs_loo(Y,pp); 
for i=1:length(pp);cali_idx_4(i,:)=fcn_getCali_human(Y,(i-1),nBins,K);end
cali_idx_4=100*cali_idx_4([2:6,1],:);

T_3=NaN(6,12);T_4=NaN(6,12);
for i=1:6
    sen=op_sen_3(i,:);fpr=op_fpr_3(i,:);ppv=op_ppv_3(i,:);cal=(cali_idx_3(i,:));    
    T_3(i,:)=[mean(sen),min(sen),max(sen),mean(fpr),min(fpr),max(fpr),mean(ppv),min(ppv),max(ppv),mean(cal),min(cal),max(cal)];  
    sen=op_sen_4(i,:);fpr=op_fpr_4(i,:);ppv=op_ppv_4(i,:);cal=(cali_idx_4(i,:));
    T_4(i,:)=[mean(sen),min(sen),max(sen),mean(fpr),min(fpr),max(fpr),mean(ppv),min(ppv),max(ppv), mean(cal),min(cal),max(cal)];
end

T_3=round(T_3);TT_3=cell(6*4,1);T_4=round(T_4);TT_4=cell(6*4,1);
pp={'Seizure','LPD','GPD','LRDA','GRDA','Other'};ss={'TPR','FPR','PPV','CAL'};
for i=1:length(pp)
    idx1=(i-1)*length(ss);
    for k=1:length(ss)
        idx2=(k-1)*(length(ss)-1);
        TT_3{idx1+k}=[pp{i},'_',ss{k},' ',num2str(T_3(i,idx2+1)),' (',num2str(T_3(i,idx2+2)),' to ',num2str(T_3(i,idx2+3)),')'];
        TT_4{idx1+k}=[pp{i},'_',ss{k},' ',num2str(T_4(i,idx2+1)),' (',num2str(T_4(i,idx2+2)),' to ',num2str(T_4(i,idx2+3)),')'];
    end
end
disp('--------------------------------------------------------------------')
disp([[{'Dataset 3'};TT_3],[{'Dataset 4'};TT_4]])
 