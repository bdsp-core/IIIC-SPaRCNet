%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 3: Maps of the Ictal-Interictal-Injury Continuum Learned by SPaRCNet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;clear;

%% set figure
f=figure('units','normalized','position',[0.1000,0.0315,0.4099,0.9333],'MenuBar','none','ToolBar','none','color','w');
ax={subplot('position',[.00,.75,.50,.25]);subplot('position',[.50,.75,.50,.25]);subplot('position',[.00,.00,.50,.25]);subplot('position',[.50,.00,.50,.25]);subplot('position',[.00,.25,1.00,.50])};

tmp=load('./Data/Figure3/figure3_input.mat');
Y=tmp.Y;Y_hat=tmp.Y_hat;Vxy=tmp.Vxy;

%% human
xlimts=[-10,25];ylimts=[-18,25];
patterns={'Seizure','LPD','GPD','LRDA','GRDA','Other'};

[~,y]=max(Y(:,[2:size(Y,2),1]),[],2);
Cs=flipud(jet(7));colors=NaN(length(y),3);
for k=1:length(patterns)
    idx_hat=find(y==k);
    colors(idx_hat,:)=repmat(Cs(k,:),length(idx_hat),1);
end
set(gcf,'CurrentAxes',ax{1});cla(ax{1})
hold(ax{1},'on');
    ss=scatter(ax{1},Vxy(:,1),Vxy(:,2),20,colors,'filled');
    alpha(ss,.2);axis equal;axis off;
    for i=1:6
        y1=12-(i-1)*2;y2=13-(i-1)*2;
        fill(ax{1},[10,10,12,12]+8,[y1,y2,y2,y1]+7,Cs(i,:),'edgecolor',Cs(i,:))
        text(ax{1},12.2+8,(y1+y2)/2+7.3,patterns{i},'fontsize',8,'verticalalignment','middle')
    end
    text(ax{1},mean(xlimts),23.5,'Human','fontsize',15,'horizontalalignment','center')
    set(ax{1},'xlim',xlimts,'ylim',ylimts)
hold(ax{1},'off');

%% SZ burden
yp_sz=Y_hat(:,2);yp_sz=(yp_sz-min(yp_sz))/(max(yp_sz)-min(yp_sz)+eps);
K=20;A=1.4;M=100;
Cs=flipud(hot(round(K*A)));Cs=Cs(round(K*(A-1))+1:end,:);colors_sz=NaN(length(yp_sz),3);
for k=1:K
    y1=(k-1)*0.05;y2=(k)*0.05;
    if k<K
        idx_sz=find(yp_sz>=y1&yp_sz<y2);
        colors_sz(idx_sz,:)=repmat(Cs(k,:),length(idx_sz),1);
    else
        idx_sz=find(yp_sz>=y1&yp_sz<=y2);
        colors_sz(idx_sz,:)=repmat(Cs(k,:),length(idx_sz),1);
    end
end
set(gcf,'CurrentAxes',ax{3});cla(ax{3})
hold(ax{3},'on');
    ss=scatter(ax{3},Vxy(:,1),Vxy(:,2),20,colors_sz,'filled');
    alpha(ss,.2);axis equal;axis off;
    
    cs=flipud(hot(M));cs=cs(round((1-1/A)*M):M,:);cs=reshape(cs,size(cs,1),1,3);dd=max(Vxy(:,2))-size(cs,1)/6;
    image(ax{3},20:20.8,(1:size(cs,1))/6+dd,cs)
    text(ax{3},21,1/6+dd,'min','fontsize',8,'verticalalignment','top','horizontalalignment','left');
    text(ax{3},21,size(cs,1)/6+dd,'max','fontsize',8,'verticalalignment','bottom','horizontalalignment','left');
    text(ax{3},mean(xlimts),23.5,'SZ burden','fontsize',15,'horizontalalignment','center')
    set(ax{3},'xlim',xlimts,'ylim',ylimts)  
hold(ax{3},'off');

%% IIIC burden
yp_iiic=sum(Y_hat(:,2:5),2);yp_iiic=(yp_iiic-min(yp_iiic))/(max(yp_iiic)-min(yp_iiic)+eps);
Cs=flipud(hot(round(K*A)));Cs=Cs(round(K*(A-1))+1:end,:);colors_iiic=NaN(length(yp_iiic),3);
for k=1:K
    y1=(k-1)*0.05;y2=(k)*0.05;
    if k<K
        idx_iiic=find(yp_iiic>=y1&yp_iiic<y2);
        colors_iiic(idx_iiic,:)=repmat(Cs(k,:),length(idx_iiic),1);
    else
        idx_iiic=find(yp_iiic>=y1&yp_iiic<=y2);
        colors_iiic(idx_iiic,:)=repmat(Cs(k,:),length(idx_iiic),1);
    end
end
set(gcf,'CurrentAxes',ax{4});cla(ax{4})
hold(ax{4},'on');
    ss=scatter(ax{4},Vxy(:,1),Vxy(:,2),20,colors_iiic,'filled');
    alpha(ss,.2);axis equal;axis off;
    cs=flipud(hot(M));cs=cs(round((1-1/A)*M):M,:);cs=reshape(cs,size(cs,1),1,3);dd=max(Vxy(:,2))-size(cs,1)/6;
    image(ax{4},20:20.8,(1:size(cs,1))/6+dd,cs)
    text(ax{4},21,1/6+dd,'min','fontsize',8,'verticalalignment','top','horizontalalignment','left');
    text(ax{4},21,size(cs,1)/6+dd,'max','fontsize',8,'verticalalignment','bottom','horizontalalignment','left');
    text(ax{4},mean(xlimts),23.5,'IIIC burden','fontsize',15,'horizontalalignment','center')  
    set(ax{4},'xlim',xlimts,'ylim',ylimts) 
hold(ax{4},'off');

%% Model uncertainty (entropy)
en_model=-sum(Y_hat.*log2(Y_hat),2);
yp=(en_model-min(en_model))/(max(en_model)-min(en_model)+eps);
Cs=flipud(hot(round(K*A)));Cs=Cs(round(K*(A-1))+1:end,:);
colors_conf=NaN(length(yp),3);
for k=1:K
    y1=(k-1)*0.05;y2=(k)*0.05;
    if k<K
        idx_conf=find(yp>=y1&yp<y2);
        colors_conf(idx_conf,:)=repmat(Cs(k,:),length(idx_conf),1);
    else
        idx_conf=find(yp>=y1&yp<=y2);
        colors_conf(idx_conf,:)=repmat(Cs(k,:),length(idx_conf),1);
    end
end
set(gcf,'CurrentAxes',ax{2});cla(ax{2})
hold(ax{2},'on');
    ss=scatter(ax{2},Vxy(:,1),Vxy(:,2),20,colors_conf,'filled');
    alpha(ss,.2);axis equal;axis off;  
    cs=flipud(hot(M));cs=cs(round((1-1/A)*M):M,:);cs=reshape(cs,size(cs,1),1,3);dd=max(Vxy(:,2))-size(cs,1)/6;
    image(ax{2},20:20.8,(1:size(cs,1))/6+dd,cs)
    text(ax{2},21,1/6+dd,'min','fontsize',8,'verticalalignment','top','horizontalalignment','left');
    text(ax{2},21,size(cs,1)/6+dd,'max','fontsize',8,'verticalalignment','bottom','horizontalalignment','left');
    text(ax{2},mean(xlimts),23.5,'Uncertainty','fontsize',15,'horizontalalignment','center');
    set(ax{2},'xlim',xlimts,'ylim',ylimts)
hold(ax{2},'off');

%% Model prediciton
[~,y_hat]=max(Y_hat(:,[2:6,1]),[],2);
Cs=flipud(jet(7));colors_hat=NaN(length(y_hat),3);
for k=1:6
    idx_hat=find(y_hat==k);
    colors_hat(idx_hat,:)=repmat(Cs(k,:),length(idx_hat),1);
end
set(gcf,'CurrentAxes',ax{5});cla(ax{5})
hold(ax{5},'on');
    ss=scatter(ax{5},Vxy(:,1),Vxy(:,2),20,colors_hat,'filled');
    alpha(ss,.1);axis equal;axis off;
    for i=1:6
        y1=12-(i-1)*1.5;y2=13-(i-1)*1.5;
        fill(ax{5},[10,10,12,12]+8,[y1,y2,y2,y1]+7,Cs(i,:),'edgecolor',Cs(i,:))
        text(ax{5},12.2+8,(y1+y2)/2+7,patterns{i},'fontsize',10)
    end   
    text(ax{5},mean(xlimts),22.5,'Model','fontsize',18,'horizontalalignment','center')  
    set(ax{5},'xlim',xlimts,'ylim',[-18,23])
hold(ax{5},'off');

%% EEG samples on the map
tmp=load('./Data/Figure3/samples.mat');
LUT=tmp.LUT;
hold(ax{5},'on');
for i=1:size(LUT,1)
    lut=LUT{i};vxy=cell2mat(lut(:,[4,5]));
    plot(ax{5},vxy(:,1),vxy(:,2),'ko',vxy(:,1),vxy(:,2),'kx','markersize',8)
    for ii=1:size(lut,1)
        text(ax{5},vxy(ii,1)+.6,vxy(ii,2)-.6,[patterns{i},'_',num2str(ii)],'horizontalalignment','left')
    end
end
hold(ax{5},'off');

print(gcf,'-r300','-dpng','./Fig3.png');
