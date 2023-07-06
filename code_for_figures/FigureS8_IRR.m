%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure S8: Additional performance metrics for SPaRCNet.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;clear;
addpath('./Callbacks/');

% set figure 
figure('units','normalized','position',[0.0000,0.0463,1.0000,0.9269],'MenuBar','none','ToolBar','none','color','w');
ax={subplot('position',[.03,.03,.43,.35]);subplot('position',[.03,.45,.43,.51]);subplot('position',[.49,.03,.24,.44]);subplot('position',[.49,.52,.24,.44]);subplot('position',[.75,.03,.24,.44]);subplot('position',[.75,.52,.24,.44])};
patterns={'Other','Seizure','LPD','GPD','LRDA','GRDA'};

% get data
tmp=load('./Data/FigureS8/FigureS8_input.mat');
Y_model=tmp.Yh;Y=tmp.Y;

% re-calibrate with post rules learnt from training set dataset#3
best_thresh=[0,0.0268,0.0016,0,0.00070,0;0,0,0.3476,0.3242,0.2916,0;0,0.0132,0,0.0754,0.0027,0;0,0.333,0.1615,0,0.2917,0;0,0.2527,0.4035,0.2862,0,0;0.4574,0,0,0.0375,0.0066,0];
y_model=fcn_getModelVotes(Y_model,best_thresh);

% IRR with CI from bootstrap 10K times
pIRR_EE=tmp.pIRR_EE;pIRR_EA=tmp.pIRR_EA;
mIRR_EE=tmp.mIRR_EE;mIRR_EA=tmp.mIRR_EA;
pIRR_EE_L=prctile(pIRR_EE,2.5);pIRR_EE_U=prctile(pIRR_EE,100-2.5);
pIRR_EA_L=prctile(pIRR_EA,2.5);pIRR_EA_U=prctile(pIRR_EA,100-2.5);
mIRR_EE_L=prctile(mIRR_EE,2.5);mIRR_EE_U=prctile(mIRR_EE,100-2.5);
mIRR_EA_L=prctile(mIRR_EA,2.5);mIRR_EA_U=prctile(mIRR_EA,100-2.5);
dpIRR=pIRR_EA-pIRR_EE;dmIRR=mIRR_EA-mIRR_EE;
dpIRR_L=prctile(dpIRR,2.5);dpIRR_U=prctile(dpIRR,100-2.5);
dmIRR_L=prctile(dmIRR,2.5);dmIRR_U=prctile(dmIRR,100-2.5);

% real CM and pIRR bars
P_human=fcn_get_pIRR_human(Y);
P_model=fcn_get_pIRR_model(Y,y_model);

pirr_ee=diag(P_human);pirr_ea=diag(P_model);

titleStr='ee-pCM (%)';
fcn_plotConfusionMx(P_human,titleStr,patterns([2:6 1]),'Purples',ax{4},1,'C');

titleStr='ea-pCM (%)';
fcn_plotConfusionMx(P_model,titleStr,patterns([2:6 1]),'Purples',ax{6},1,'D');

% real CM and mIRR bars
M_human=fcn_get_mIRR_human_loo(Y);
M_model=fcn_get_mIRR_model(Y,y_model);
mirr_ee=diag(M_human);mirr_ea=diag(M_model);

titleStr='ee-mCM (%)';
fcn_plotConfusionMx(M_human,titleStr,patterns([2:6,1]),'Purples',ax{3},2,'E');

titleStr='ea-mCM (%)';
fcn_plotConfusionMx(M_model,titleStr,patterns([2:6,1]),'Purples',ax{5},2,'F');

% IRR bars
set(gcf,'CurrentAxes',ax{2});cla(ax{2})
hold(ax{2},'on');
    cc=brewermap(4,'RdBu');cc=cc([2,1,3,4],:);
    X=[pirr_ee,pirr_ea,mirr_ee,mirr_ea];
    b=bar(ax{2},X,'FaceColor','flat');
    for k=1:size(X,2)
        b(k).CData=cc(k,:);
    end 
    xdata=[0.7273,1.7273,2.7273,3.7273,4.7273,5.7273;0.9091,1.9091,2.9091,3.9091,4.9091,5.9091;1.0909,2.0909,3.0909,4.0909,5.0909,6.0909;1.2727,2.2727,3.2727,4.2727,5.2727,6.2727];
     
    pp={'Seizure','LPD','GPD','LRDA','GRDA','Other'};
    set(ax{2},'xtick',1:6,'xticklabel',pp,'fontsize',12)
    
    dd=0.03;
    for i=1:6
        plot(ax{2},[xdata(1,i) xdata(1,i)],[pIRR_EE_L(i),pIRR_EE_U(i)],'k-','linewidth',1)
        plot(ax{2},[xdata(1,i)-dd xdata(1,i)+dd],[pIRR_EE_L(i),pIRR_EE_L(i)],'k-','linewidth',1)
        plot(ax{2},[xdata(1,i)-dd xdata(1,i)+dd],[pIRR_EE_U(i),pIRR_EE_U(i)],'k-','linewidth',1)
        
        plot(ax{2},[xdata(2,i) xdata(2,i)],[pIRR_EA_L(i),pIRR_EA_U(i)],'k-','linewidth',1)
        plot(ax{2},[xdata(2,i)-dd xdata(2,i)+dd],[pIRR_EA_L(i),pIRR_EA_L(i)],'k-','linewidth',1)
        plot(ax{2},[xdata(2,i)-dd xdata(2,i)+dd],[pIRR_EA_U(i),pIRR_EA_U(i)],'k-','linewidth',1)
        
        plot(ax{2},[xdata(3,i) xdata(3,i)],[mIRR_EE_L(i),mIRR_EE_U(i)],'k-','linewidth',1)
        plot(ax{2},[xdata(3,i)-dd xdata(3,i)+dd],[mIRR_EE_L(i),mIRR_EE_L(i)],'k-','linewidth',1)
        plot(ax{2},[xdata(3,i)-dd xdata(3,i)+dd],[mIRR_EE_U(i),mIRR_EE_U(i)],'k-','linewidth',1)
        
        plot(ax{2},[xdata(4,i) xdata(4,i)],[mIRR_EA_L(i),mIRR_EA_U(i)],'k-','linewidth',1)
        plot(ax{2},[xdata(4,i)-dd xdata(4,i)+dd],[mIRR_EA_L(i),mIRR_EA_L(i)],'k-','linewidth',1)
        plot(ax{2},[xdata(4,i)-dd xdata(4,i)+dd],[mIRR_EA_U(i),mIRR_EA_U(i)],'k-','linewidth',1)
    end 
    legend(b,'ee-pIRR','ea-pIRR','ee-mIRR','ea-mIRR')
    legend('boxoff')
    ylim([0,1])
    text(ax{2},0.1,1.03,'A','fontsize',25)
hold(ax{2},'off');

% diff-IRR
set(gcf,'CurrentAxes',ax{1});cla(ax{1})
hold(ax{1},'on');
    cc=brewermap(2,'RdBu');
    X =[pirr_ea-pirr_ee,mirr_ea- mirr_ee];
    b=bar(ax{1},X,'FaceColor','flat');
    for k=1:size(X,2)
        b(k).CData=cc(k,:);
    end
    xdata=[0.8571,1.8571,2.8571,3.8571,4.8571,5.8571;1.1429,2.1429,3.1429,4.1429,5.1429,6.1429];
    set(ax{1},'xtick',1:6,'xticklabel',pp,'fontsize',12)
    
    for i=1:6
        plot(ax{1},[xdata(1,i) xdata(1,i)],[dpIRR_L(i),dpIRR_U(i)],'k-','linewidth',1)
        plot(ax{1},[xdata(1,i)-dd xdata(1,i)+dd],[dpIRR_L(i),dpIRR_L(i)],'k-','linewidth',1)
        plot(ax{1},[xdata(1,i)-dd xdata(1,i)+dd],[dpIRR_U(i),dpIRR_U(i)],'k-','linewidth',1)
        
        plot(ax{1},[xdata(2,i)    xdata(2,i)],[dmIRR_L(i),dmIRR_U(i)],'k-','linewidth',1)
        plot(ax{1},[xdata(2,i)-dd xdata(2,i)+dd],[dmIRR_L(i),dmIRR_L(i)],'k-','linewidth',1)
        plot(ax{1},[xdata(2,i)-dd xdata(2,i)+dd],[dmIRR_U(i),dmIRR_U(i)],'k-','linewidth',1)
    end  
    legend(b,'ea-ee pIRR','ea-ee mIRR')
    legend('boxoff')
    ylim([-0.15,0.15])
    
    text(ax{1},0.1,0.15,'B','fontsize',25)
    text(ax{1},.7, 0.1,'ea>ee','fontsize',15)
    text(ax{1},.7,-0.1,'ea<ee','fontsize',15) 
hold(ax{1},'off');

print(gcf,'-r300','-dpng', './FigS8.png');
