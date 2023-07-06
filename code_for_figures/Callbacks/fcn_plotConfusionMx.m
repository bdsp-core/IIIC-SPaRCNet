function fcn_plotConfusionMx(Mx,titleStr,labels,colorStr,ax,jj,figure_idx)
    set(gcf,'CurrentAxes',ax);cla(ax)
    hold(ax,'on');
        x=1:6;
        imagesc(ax,x,x,Mx,[0,1])
        for i=0:7
           xx=[i,i]-0.5;yy=[-1,7];
           plot(ax,xx,yy,'k',yy,xx,'k');
        end
        for i=1:6
            for j=1:6
                n=round(Mx(i,j)*100);
                if n>10;xx=j-.2;else;xx=j - 0.1;end;yy=i;
                if n<=50;text(ax,xx,yy,num2str(n),'color','k','fontsize',20);else;text(ax,xx,yy,num2str(n),'color','w','fontsize',20);end
            end
        end
        axis ij;axis square;box off
        colormap(ax,brewermap([],colorStr))
        set(ax,'TickLength',[0,0],'xtick',1:6,'ytick',1:6,'xticklabels',labels,'yticklabels',labels,'fontsize',12)
        if jj == 1
            set(ax,'xtick',1:6,'ytick',1:6,'xticklabels',[],'yticklabels',labels,'fontsize',12)
        end
        text(ax,0,0.25,figure_idx,'fontsize',25)
        xlim([.5 6.5]);ylim([.5 6.5])
        title(titleStr);
    hold(ax,'off');
end
