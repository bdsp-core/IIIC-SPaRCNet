function fcn_plotROC(sp,ax,cc,op_fpr,op_sen,fpr_M,sen_M,auroc_M,fpr_L,sen_L,auroc_L,fpr_U,sen_U,auroc_U,EUROC_T,EUROC_L,EUROC_U)
    patterns={'Seizure','LPD','GPD','LRDA','GRDA','Other'};
    set(gcf,'CurrentAxes',ax{sp});cla(ax{sp})
    hold(ax{sp},'on');
        xx=[fpr_L,fliplr(fpr_U)];yy=[sen_L,fliplr(sen_U)];
        patch(ax{sp},xx,yy,'b', 'facealpha',.3,'edgecolor', 'none', 'facecolor',cc);
        plot(ax{sp},fpr_M,sen_M,'-', 'color',cc,'linewidth',2);
        scatter(ax{sp},op_fpr,op_sen,50,[0,0,0],'filled');idx=[];
        for kk=1:length(op_fpr)
            op_sen_kk=op_sen(kk);op_fpr_kk=op_fpr(kk);
            [~,ii]=min(abs(op_fpr_kk-fpr_M));
            if  op_sen_kk<=sen_M(ii) 
                idx=[idx;kk];
            end
        end
        scatter(ax{sp},op_fpr(idx),op_sen(idx),50,[0.5 0.5 0.5],'filled');
        text(ax{sp},20,20,['AUROC: ',num2str(round(auroc_M*1000)/10),' (',num2str(round(auroc_L*1000)/10),', ',num2str(round(auroc_U*1000)/10),')%', '\newlineEUROC: ',num2str(round(EUROC_T*1000)/10),' (',num2str(round(EUROC_L*1000)/10),', ',num2str(round(EUROC_U*1000)/10),')%'],'fontsize',12);
        set(ax{sp},'xtick',0:20:100,'ytick',0:20:100,'xlim',[0,100],'ylim',[0,100],'fontsize',12);
        xlabel('FPR');ylabel('TPR');axis square;grid on;box off;
        title(patterns{sp},'fontsize',20);
    hold(ax{sp},'off');
end
 
 