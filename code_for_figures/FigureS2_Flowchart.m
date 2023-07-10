%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure S2: Flowchart on data splits.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;clear;
 
dataDir='./Data/FigureS2/';

%% Dataset A: all Real
tmp=load([dataDir,'datasetA.mat']);keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6 1])';
str1=['Dataset A: ',num2str(length(raters)),' raters  ',num2str(length(pats)),' patients  ',num2str(length(y)),' samples'];
str2=['SZ    ',num2str(c(1)) ];str3=['LPD   ',num2str(c(2)) ];str4=['GPD   ',num2str(c(3)) ];str5=['LRDA  ',num2str(c(4)) ];str6=['GRDA  ',num2str(c(5)) ];str7=['Other ',num2str(c(6)) ];
disp('-----------------------------------------------------')
disp(str1);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)

%% Dataset B: 1+2 Real (>3)
tmp=load([dataDir,'datasetB.mat']);keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6 1])';
str1=['Dataset B: ',num2str(length(raters)),' raters  ',num2str(length(pats)),' patients  ',num2str(length(y)),' samples'];
str2=['SZ    ',num2str(c(1)) ];str3=['LPD   ',num2str(c(2)) ];str4=['GPD   ',num2str(c(3)) ];str5=['LRDA  ',num2str(c(4)) ];str6=['GRDA  ',num2str(c(5)) ];str7=['Other ',num2str(c(6)) ];
disp('-----------------------------------------------------')
disp(str1);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
%% Dataset C: 3+4 Real (>10)
tmp=load([dataDir,'datasetC.mat']);keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6 1])';
str1=['Dataset C: ',num2str(length(raters)),' raters  ',num2str(length(pats)),' patients  ',num2str(length(y)),' samples'];
str2=['SZ    ',num2str(c(1)) ];str3=['LPD   ',num2str(c(2)) ];str4=['GPD   ',num2str(c(3)) ];str5=['LRDA  ',num2str(c(4)) ];str6=['GRDA  ',num2str(c(5)) ];str7=['Other ',num2str(c(6)) ];
disp('-----------------------------------------------------')
disp(str1);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
%% Dataset D: 3+4 Real+SP-spread
tmp=load([dataDir,'datasetD.mat']);keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6 1])';
str1=['Dataset D: ',num2str(length(raters)),' raters  ',num2str(length(pats)),' patients  ',num2str(length(y)),' samples'];
str2=['SZ    ',num2str(c(1)) ];str3=['LPD   ',num2str(c(2)) ];str4=['GPD   ',num2str(c(3)) ];str5=['LRDA  ',num2str(c(4)) ];str6=['GRDA  ',num2str(c(5)) ];str7=['Other ',num2str(c(6)) ];
disp('-----------------------------------------------------')
disp(str1);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
%% Dataset 1: Real+SP-spread
tmp=load([dataDir,'dataset1.mat']);keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6 1])';
str1=['Dataset 1: ',num2str(length(raters)),' raters  ',num2str(length(pats)),' patients  ',num2str(length(y)),' samples'];
str2=['SZ    ',num2str(c(1)) ];str3=['LPD   ',num2str(c(2)) ];str4=['GPD   ',num2str(c(3)) ];str5=['LRDA  ',num2str(c(4)) ];str6=['GRDA  ',num2str(c(5)) ];str7=['Other ',num2str(c(6)) ];
disp('-----------------------------------------------------')
disp(str1);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
%% Dataset 2: Real+SP-spread+UMAP-spread
tmp=load([dataDir,'dataset2.mat']);keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6 1])';
str1=['Dataset 2: ',num2str(length(raters)),' raters  ',num2str(length(pats)),' patients  ',num2str(length(y)),' samples'];
str2=['SZ    ',num2str(c(1)) ];str3=['LPD   ',num2str(c(2)) ];str4=['GPD   ',num2str(c(3)) ];str5=['LRDA  ',num2str(c(4)) ];str6=['GRDA  ',num2str(c(5)) ];str7=['Other ',num2str(c(6)) ];
disp('-----------------------------------------------------')
disp(str1);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
%% Dataset 3: Real+SP-spread
tmp=load([dataDir,'dataset3.mat']);keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6 1])';
str1=['Dataset 3: ',num2str(length(raters)),' raters  ',num2str(length(pats)),' patients  ',num2str(length(y)),' samples'];
str2=['SZ    ',num2str(c(1)) ];str3=['LPD   ',num2str(c(2)) ];str4=['GPD   ',num2str(c(3)) ];str5=['LRDA  ',num2str(c(4)) ];str6=['GRDA  ',num2str(c(5)) ];str7=['Other ',num2str(c(6)) ];
disp('-----------------------------------------------------')
disp(str1);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
 
%% Dataset 4: Real+SP-spread
tmp=load([dataDir,'dataset4.mat']);keys=tmp.keys;Y=tmp.Y;pats=cell(size(Y,1),1);for i=1:size(Y,1);x=split(keys{i},'_');pats{i}=x{2};end;pats=unique(pats);
raters=find(sum(~isnan(Y),1)>0);y=mode(Y,2);c=hist(y,0:5);c=c([2:6 1])';
str1=['Dataset 4: ',num2str(length(raters)),' raters  ',num2str(length(pats)),' patients  ',num2str(length(y)),' samples'];
str2=['SZ    ',num2str(c(1)) ];str3=['LPD   ',num2str(c(2)) ];str4=['GPD   ',num2str(c(3)) ];str5=['LRDA  ',num2str(c(4)) ];str6=['GRDA  ',num2str(c(5)) ];str7=['Other ',num2str(c(6)) ];
disp('-----------------------------------------------------')
disp(str1);disp(str2);disp(str3);disp(str4);disp(str5);disp(str6);disp(str7)
