function fcn_plotPR(sp,ax,c,op_sen,op_ppv,sen_M,ppv_M,auprc_M,sen_L,ppv_L,auprc_L,sen_U,ppv_U,auprc_U,EUPRC_T,EUPRC_L,EUPRC_U)
    labels={'Seizure','LPD','GPD','LRDA','GRDA','Other'}; 
    set(gcf,'CurrentAxes',ax{sp});cla(ax{sp})
    hold(ax{sp},'on');
        xx=[sen_L,fliplr(sen_U)];yy=[ppv_L,fliplr(ppv_U)]; 
        patch(ax{sp},xx,yy,'b','facealpha',.3,'edgecolor','none','facecolor',c); 
        plot(ax{sp},sen_M,ppv_M,'-','color',c,'linewidth',2); 

        scatter(ax{sp},op_sen,op_ppv,50,[0 0 0],'filled','marker','v');
        idx=[];
        for kk=1:length(op_sen)
            op_sen_kk=op_sen(kk);op_ppv_kk=op_ppv(kk); 
            [~,ii]=min(abs(op_sen_kk-sen_M));
            if  op_ppv_kk<=ppv_M(ii)
                idx=[idx;kk];
            end
        end
        scatter(ax{sp},op_sen(idx),op_ppv(idx),50,[0.5,0.5,0.5],'filled','marker','v');
        if sp==1
            text(ax{sp},20,80,['AUPRC: ',num2str(round(auprc_M*1000)/10),' (',num2str(round(auprc_L*1000)/10),', ',num2str(round(auprc_U*1000)/10),')%','\newlineEUPRC: ',num2str(round(EUPRC_T*1000)/10),' (',num2str(round(EUPRC_L*1000)/10),', ',num2str(round(EUPRC_U*1000)/10),')%'],'fontsize',12);  
        else
            text(ax{sp},20,20,['AUPRC: ',num2str(round(auprc_M*1000)/10),' (',num2str(round(auprc_L*1000)/10),', ',num2str(round(auprc_U*1000)/10),')%','\newlineEUPRC: ',num2str(round(EUPRC_T*1000)/10),' (',num2str(round(EUPRC_L*1000)/10),', ',num2str(round(EUPRC_U*1000)/10),')%'],'fontsize',12); 
        end
        set(ax{sp},'xtick',0:20:100,'ytick',0:20:100,'xlim',[0,100],'ylim',[0,100],'fontsize',12);
        xlabel('TPR'); ylabel('PPV'); axis square; grid on; box off
        title(labels{sp},'fontsize',20); 
    hold(ax{sp},'off');
end

 
 