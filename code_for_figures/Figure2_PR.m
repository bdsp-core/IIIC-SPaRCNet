%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 2: Evaluation of model performance relative to experts: PR curves
% plot the true human operating points 
% report the EUPRC using data with K=1E4 bootstrap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;clear;
 
addpath('./Callbacks/');
load('./Data/Figure1/figure1_input.mat')

figure('units','normalized','position',[0.0615,0.0889,0.7703,0.8204],'MenuBar','none','ToolBar','none','color','w');
ax={subplot('position',[.050,.564,.250,.350]);subplot('position',[.370,.564,.250,.350]);subplot('position',[.690,.564,.250,.350]);subplot('position',[.050,.080,.250,.350]);subplot('position',[.370,.080,.250,.350]);subplot('position',[.690,.080,.250,.350])};

p=[1:5,0];nClass=length(p);
cc=[0.635,0.078,0.184;0.850,0.325,0.098;0.929,0.694,0.125;0.466,0.674,0.188;0.301,0.745,0.933;0.000,0.447,0.741];

for i=1:nClass
    fcn_plotPR(i,ax,cc(i,:),op_sen_T(i,:),op_ppv_T(i,:),sen_Median(i,:),ppv_Median(i,:),auprc_Median(i),sen_L(i,:),ppv_L(i,:),auprc_L(i),sen_U(i,:),ppv_U(i,:),auprc_U(i),EUPRC_Median(i),EUPRC_L(i),EUPRC_U(i));
end
print(gcf,'-r300','-dpng','Fig2.png');
