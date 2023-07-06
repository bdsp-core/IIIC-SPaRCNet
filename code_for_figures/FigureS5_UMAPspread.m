%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure S5. Creation of pseudo-labels via “label spreading” in Steps 3-4 
% of the model development procedure for SPaRCNet.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;clear;
addpath('./Callbacks/');

%% set figure
figure('units','normalized','position',[0.0,0.1,0.9,0.8],'MenuBar','none','ToolBar','none','color','w');
ax={subplot('position',[0/6,5/6,1/6,1/6]);subplot('position',[1/6,5/6,1/6,1/6]);subplot('position',[2/6,5/6,1/6,1/6]);subplot('position',[3/6,5/6,1/6,1/6]);subplot('position',[4/6,5/6,1/6,1/6]);subplot('position',[5/6,5/6,1/6,1/6]);subplot('position',[5/6,4/6,1/6,1/6]);subplot('position',[5/6,3/6,1/6,1/6]);subplot('position',[5/6,2/6,1/6,1/6]);subplot('position',[5/6,1/6,1/6,1/6]);subplot('position',[5/6,0/6,1/6,1/6]);subplot('position',[4/6,0/6,1/6,1/6]);subplot('position',[3/6,0/6,1/6,1/6]);subplot('position',[2/6,0/6,1/6,1/6]);subplot('position',[1/6,0/6,1/6,1/6]);subplot('position',[0/6,0/6,1/6,1/6]);subplot('position',[0/6,1/6,1/6,1/6]);subplot('position',[0/6,2/6,1/6,1/6]);subplot('position',[0/6,3/6,1/6,1/6]);subplot('position',[0/6,4/6,1/6,1/6]);subplot('position',[1/6,1/6,4/6 4/6])};
  
%% UMAP and labels
tmp=load('./Data/FigureS5/FigureS5_input.mat');
Vxy=tmp.Vxy;Y_real=tmp.Y;Y_pseu=tmp.Y_spread;IDX=tmp.idx;
y_real=mode(Y_real,2);y_real(y_real==0)=6;
y_pseu=mode(Y_pseu,2);y_pseu(y_pseu==0)=6;

xlimts=[-15,60];ylimts=[-18,25];
Cs=flipud(jet(7));
colors_real=0.8*ones(length(y_real),3);
colors_pseu=0.8*ones(length(y_real),3);

for k=1:6
    idx=find(y_real==k);colors_real(idx,:)=repmat(Cs(k,:),length(idx),1);
    idx=find(y_pseu==k);colors_pseu(idx,:)=repmat(Cs(k,:),length(idx),1);
end

set(gcf,'CurrentAxes',ax{21});cla(ax{21})
hold(ax{21},'on');
    text(ax{21},xlimts(1)+5,ylimts(2)-3,'A','fontsize',20);
    text(ax{21},5,25/2+9,'Before','fontsize',15);   
    text(ax{21},41,25/2+9,'After','fontsize',15);   
    ss1=scatter(ax{21},Vxy(:,1),Vxy(:,2),20,colors_real,'filled');alpha(ss1,.2)
    ss2=scatter(ax{21},Vxy(:,1)+35,Vxy(:,2),20,colors_pseu,'filled');alpha(ss2,.2)
    
    patterns={'Seizure','LPD','GPD','LRDA','GRDA','Other'};
    for i=1:6
        y1=12-(i-1)*2;y2=13-(i-1)*2;
        fill(ax{21},[10,10,12,12]+8,[y1,y2,y2,y1]+7,Cs(i,:),'edgecolor',Cs(i,:))
        text(ax{21},12.2+8,(y1+y2)/2+7.3,patterns{i},'fontsize',10,'verticalalignment','middle')
    end    
    set(ax{21},'xlim',xlimts,'ylim',ylimts) 
    axis off;
hold(ax{21},'off');

%% individuals
subIDX={'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
for jj=1:length(IDX)
    jj1=IDX(jj);   
    y_real=Y_real(:,jj1);y_real(y_real==0)=6;
    y_pseu=Y_pseu(:,jj1);y_pseu(y_pseu==0)=6;

    colors_real=NaN(length(y_real),3);
    colors_pseu=NaN(length(y_real),3);
    for k=1:6
        idx=find(y_real==k);colors_real(idx,:)=repmat(Cs(k,:),length(idx),1);
        idx=find(y_pseu==k);colors_pseu(idx,:)=repmat(Cs(k,:),length(idx),1);
    end
    set(gcf,'CurrentAxes',ax{jj});cla(ax{jj})
    hold(ax{jj},'on');
        if jj == 1
            text(ax{jj},xlimts(1)+1,ylimts(2)-4,subIDX{jj},'fontsize',20)
        end
        idx1=find(~isnan(y_real));idx0=find(isnan(y_real));
        ss0=scatter(ax{jj},Vxy(idx0,1),Vxy(idx0,2),20,repmat([0.8,0.8,0.8],length(idx0),1),'filled');alpha(ss0,.2);
        ss1=scatter(ax{jj},Vxy(idx1,1),Vxy(idx1,2),20,colors_real(idx1,:),'filled');alpha(ss1,.05);
        ss2=scatter(ax{jj},Vxy(:,1)+35,Vxy(:,2),20,colors_pseu,'filled');alpha(ss2,.2)
        text(ax{jj},20,25/2+10,['E',repmat('0',1,2-length(num2str(jj))),num2str(jj)],'fontsize',10)       
        set(ax{jj},'xlim',xlimts,'ylim',ylimts);axis off;
    hold(ax{jj},'off');
    drawnow
end

%%
print(gcf,'-r300','-dpng', './FigS5.png');