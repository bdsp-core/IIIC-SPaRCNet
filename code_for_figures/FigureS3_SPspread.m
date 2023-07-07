%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure S3: Samples belonging to the same stationary period (SP) are 
% assigned the same label.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;close all;clear;
addpath('./Callbacks/');

%% set figure
f=figure('units','normalized','position',[0.1094,0.0472,0.8531,0.8574],'MenuBar','none','ToolBar','none','color','w');
ax={subplot('position',[.03,.86,.96,.12]);subplot('position',[.03,.74,.96,.12]);subplot('position',[.03,.62,.96,.12]);subplot('position',[.03,.50,.96,.12]);subplot('position',[.03,.40,.96,.10]);subplot('position',[.03,.03,.30,.34]);subplot('position',[.36,.03,.30,.34]);subplot('position',[.69,.03,.30,.34])};   

tmp=load('./Data/FigureS3/FigureS3_input');
seg_t1=tmp.seg_t1;seg_t2=tmp.seg_t2;seg_tc=tmp.seg_tc;
thr_cp=.3;ss=tmp.ss;S_x=tmp.S_x;S_y=tmp.S_y;S_data=tmp.S_data;

%% qEEG
[icp,P,iscp,iscpc]=fcn_cpd(ss,thr_cp);
col=[-10,25];colormap jet;spatialRegs={'LL','RL','LP','RP'};
for i=1:4
    set(f,'CurrentAxes',ax{i});cla(ax{i})
    spec=S_data{i,1};
    imagesc(ax{i},S_x,S_y,pow2db(spec),col);  
    axis(ax{i},'xy'); 
    xx=get(ax{i},'yticklabel');
    text(ax{i},S_x(1),18,spatialRegs{i},'fontsize',15,'color','w','fontweight','bold')
    set(ax{i},'xtick',[],'box','on','yticklabel',xx)
    ylabel(ax{i},'Freq (Hz)')
end

set(f,'CurrentAxes',ax{5});cla(ax{5})
stimes=S_x;tc=length(stimes)/2+1;
hold(ax{5},'on')
    a=min(P)-5;b=max(P)+1;
    plot(ax{5},stimes,P,'g-','linewidth',2)
    set(ax{5},'xtick',[])
    for icpd=1:length(icp)-1
        aa_=icp(icpd);bb_=icp(icpd+1);   
        if icpd==10
            cc_=round((aa_+bb_)/2);
            plot(ax{5},[stimes(cc_),stimes(cc_)],[a b],'r-.','linewidth',1)      
            text(ax{5},stimes(cc_),a,'t_C','verticalalignment','top','horizontalalignment','center')
            text(ax{5},stimes(round((bb_+cc_)/2)),a,'t_2','verticalalignment','top','horizontalalignment','left')
            text(ax{5},stimes(round((aa_+cc_)/2)),a,'t_1','verticalalignment','top','horizontalalignment','left')
            plot(ax{5},stimes(round((bb_+cc_)/2)),a+3,'rv','markersize',10,'markerfacecolor','r')
            plot(ax{5},stimes(round((aa_+cc_)/2)),a+3,'rv','markersize',10,'markerfacecolor','r')
        end    
        if icpd>1
            plot(ax{5},[stimes(aa_),stimes(aa_)],[a b],'m-.','linewidth',1)
        end  
        cpd_mean=mean(P(aa_:bb_));
        plot(ax{5},[stimes(aa_),stimes(bb_)],[cpd_mean,cpd_mean],'b-','linewidth',1)      
    end
    xlim([stimes(1) stimes(end)]);ylim([a b]);ylabel('Power (dB)');box on
   
    plot(ax{5},[stimes(tc),stimes(tc)],[a b],'b--','linewidth',1) 
    text(ax{5},stimes(1),a,'00:00:00','verticalalignment','top','horizontalalignment','left')      
    text(ax{5},stimes(end),a,'00:10:00','verticalalignment','top','horizontalalignment','right')  
hold(ax{5},'off')

%% EEG
fcn_plotEEG(f,ax{6},seg_t1,'L-Bipolar',1,'t_1')
fcn_plotEEG(f,ax{7},seg_tc,'L-Bipolar',0,'t_C')
fcn_plotEEG(f,ax{8},seg_t2,'L-Bipolar',0,'t_2')

print(gcf,'-r300','-dpng', './FigS3.png');
