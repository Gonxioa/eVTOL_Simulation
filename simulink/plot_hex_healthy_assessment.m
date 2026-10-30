function figures=plot_hex_healthy_assessment(r,folder,m,show_figures)
%PLOT_HEX_HEALTHY_ASSESSMENT Scalar overviews and three independent axis figures.
if nargin<4, show_figures=true; end
t=r.time; s=r.state; ref=r.reference; P=r.parameters;
pe=s(:,1:3)-ref(:,1:3); ve=s(:,4:6)-ref(:,4:6);
pn=vecnorm(pe,2,2); vn=vecnorm(ve,2,2);
visible='off'; if show_figures, visible='on'; end
figures=gobjects(1,5);
figures(1)=make_figure([m.task_name '：跟踪总览'],visible,[60 50 1300 850]);
tiledlayout(figures(1),2,2,'TileSpacing','compact','Padding','loose');
nexttile;
plot3(ref(:,1),ref(:,2),ref(:,3),'k--','LineWidth',2,'DisplayName','期望轨迹'); hold on;
plot3(s(:,1),s(:,2),s(:,3),'LineWidth',1.1,'DisplayName','实际轨迹');
scatter3(s(1,1),s(1,2),s(1,3),35,[.1 .6 .3],'filled','DisplayName','起点');
scatter3(s(end,1),s(end,2),s(end,3),35,[.8 .2 .2],'filled','DisplayName','终点');
xlabel('x（m）'); ylabel('y（m）'); zlabel('z（m）');
axis equal; grid on; view(35,24); title('三维轨迹总览'); legend('Location','best','FontSize',8);
limits=zeros(3,2);
for j=1:3
    values=[s(:,j);ref(:,j)]; center=(min(values)+max(values))/2;
    width=max(.3,1.16*(max(values)-min(values)));
    limits(j,:)=center+[-.5 .5]*width;
end
xlim(limits(1,:)); ylim(limits(2,:)); zlim(limits(3,:));
if P.task_family==0
    xlim(P.task_origin(1)+[-.10 .10]); ylim(P.task_origin(2)+[-.10 .10]);
    zlim(P.task_origin(3)+[-.10 .10]);
end
nexttile; plot(t,100*pn,'LineWidth',1.2); time_axis(t,m);
title(sprintf('三维位置误差｜RMSE %.2f cm，峰值 %.2f cm',100*m.position_rmse_m,100*m.position_peak_m));
ylabel('位置误差范数（cm）');
nexttile; plot(t,vn,'LineWidth',1.2); time_axis(t,m);
title(sprintf('三维速度误差｜RMSE %.3f m/s',m.velocity_rmse_m_s)); ylabel('速度误差范数（m/s）');
nexttile; hold on;
plot(t,t,'k--','LineWidth',1.1,'DisplayName','名义任务时间');
plot(t,r.recovery_reference_audit(:,1),'LineWidth',1.2,'DisplayName','实际路径进度时间');
if P.task_family~=0, yline(P.motion_end,':','DisplayName','名义运动终点'); end
time_axis(t,m);
if P.task_family==0, title('任务进度｜悬停无运动阶段');
else, title('任务进度｜竖虚线：实际运动结束'); end
ylabel('路径进度时间（s）'); legend('Location','northwest','FontSize',8);
sgtitle(sprintf('%s｜健康工况｜轨迹与三维误差总览',m.task_name),'FontWeight','bold');
save_figure(figures(1),folder,'tracking');

figures(2)=make_figure([m.task_name '：稳定性总览'],visible,[70 60 1300 850]);
tiledlayout(figures(2),2,2,'TileSpacing','compact','Padding','loose');
nexttile; hold on;
qd=r.control_audit(:,28:31);
tilts=acosd(min(max(1-2*(s(:,7).^2+s(:,8).^2),-1),1));
desired_tilt=acosd(min(max(1-2*(qd(:,1).^2+qd(:,2).^2),-1),1));
plot(t,tilts,'LineWidth',1.2,'DisplayName','实际倾角真值');
plot(t,desired_tilt,'--','LineWidth',.9,'DisplayName','期望倾角指令');
time_axis(t,m); title(sprintf('总倾角｜实际峰值 %.2f°',m.maximum_tilt_deg));
ylabel('倾角（°）'); legend('Location','best','FontSize',8);
nexttile;
plot(t,100*pn,'LineWidth',1.1); hold on;
yline(100*m.position_band_m,'--','5 cm 观察带','Color',[.75 .25 .2],...
    'LabelHorizontalAlignment','left','HandleVisibility','off');
time_axis(t,m);
if isfinite(m.motion_finished_at_s), xlim([max(0,m.motion_finished_at_s-1),t(end)]); end
if P.task_family==0
    title('悬停位置保持｜5 cm 观察带');
elseif isfinite(m.position_band_wait_s)
    title(sprintf('运动结束后位置保持｜入带等待 %.2f s',m.position_band_wait_s));
else
    title('运动结束后位置保持｜未观察到足够长的持续入带');
end
ylabel('三维位置误差范数（cm）');
nexttile;
plot(t,100*r.motor_command/P.omega_max,'LineWidth',.8); hold on;
yline(100,'r--','指令上限','HandleVisibility','off','LabelHorizontalAlignment','left');
yline(0,'k:','HandleVisibility','off');
time_axis(t,m); ylim([-3 105]); title('六台电机转速指令占上限的比例');
ylabel('转速指令 / 上限（%）'); legend(compose('%d号',1:6),'Location','eastoutside','FontSize',8);
nexttile;
bound=zeros(size(r.motor_command)); tol=m.saturation_tolerance_rad_s;
bound(r.motor_command<=P.omega_min+tol)=-1; bound(r.motor_command>=P.omega_max-tol)=1;
imagesc(t,1:6,bound'+1); set(gca,'YDir','normal');
colormap(gca,[.2 .45 .8;.95 .95 .95;.85 .2 .15]); clim([-.5 2.5]);
cb=colorbar; cb.Ticks=[0 1 2]; cb.TickLabels={'下限','未触限','上限'};
yticks(1:6); ylabel('电机编号'); xlabel('Time (s)');
title(sprintf('指令触限状态｜时间占比 %.3f%%',m.any_motor_saturation_percent));
sgtitle(sprintf('%s｜健康工况｜总倾角与执行器余量',m.task_name),'FontWeight','bold');
save_figure(figures(2),folder,'stability');

% Every Cartesian component is plotted in its own figure, including Euler
% angles, true body rates and their rolling RMS. No data are smoothed for display.
axis_ids={'x','y','z'};
axis_names={'x','y','z'};
attitude_names={'滚转角','俯仰角','偏航角'};
rate_names={'滚转角速度 p','俯仰角速度 q','偏航角速度 r'};
actual_angles=euler_angles(s(:,7:10)); desired_angles=euler_angles(qd);
window=max(1,round(1/P.export_dt));
rolling=sqrt(movmean(s(:,11:13).^2,[window-1 0],1));
for j=1:3
    id=axis_ids{j}; f=make_figure([m.task_name '：' id],visible,[80 35 1280 1170]);
    figures(2+j)=f; set(f,'Tag',['healthy_axis_' id]);
    tiledlayout(f,4,2,'TileSpacing','compact','Padding','loose');
    nexttile;
    plot(t,s(:,j),'LineWidth',1.1,'DisplayName','实际位置','Tag',['position_actual_' id]); hold on;
    plot(t,ref(:,j),'--','LineWidth',1,'DisplayName','期望位置','Tag',['position_reference_' id]);
    time_axis(t,m); title([axis_names{j} '：位置真值与期望']); ylabel('位置（m）'); legend('Location','best');
    nexttile;
    plot(t,100*pe(:,j),'LineWidth',1,'Tag',['position_error_' id]); time_axis(t,m);
    title(sprintf('%s位置误差｜RMSE %.3f cm',axis_names{j},100*m.position_axis_rmse_m(j)));
    ylabel('位置误差（cm）');
    nexttile;
    plot(t,s(:,j+3),'LineWidth',1.1,'DisplayName','实际速度','Tag',['velocity_actual_' id]); hold on;
    plot(t,ref(:,j+3),'--','LineWidth',1,'DisplayName','期望速度','Tag',['velocity_reference_' id]);
    time_axis(t,m); title([axis_names{j} '：速度真值与期望']); ylabel('速度（m/s）'); legend('Location','best');
    nexttile;
    plot(t,ve(:,j),'LineWidth',1,'Tag',['velocity_error_' id]); time_axis(t,m);
    title(sprintf('%s速度误差｜RMSE %.4f m/s',axis_names{j},m.velocity_axis_rmse_m_s(j)));
    ylabel('速度误差（m/s）');
    nexttile;
    plot(t,actual_angles(:,j),'LineWidth',1.1,'DisplayName','实际角度真值','Tag',['euler_actual_' id]); hold on;
    plot(t,desired_angles(:,j),'--','LineWidth',.8,'DisplayName','期望角度指令','Tag',['euler_reference_' id]);
    time_axis(t,m); title([attitude_names{j} '：实际与期望']); ylabel('角度（°）'); legend('Location','best');
    nexttile;
    plot(t,r.control_audit(:,31+j),'--','LineWidth',.7,'DisplayName','角速度指令','Tag',['rate_reference_' id]); hold on;
    plot(t,s(:,10+j),'LineWidth',1.1,'DisplayName','真实机体角速度','Tag',['rate_actual_' id]);
    time_axis(t,m); title([rate_names{j} '：真值与指令']); ylabel('角速度（rad/s）'); legend('Location','best');
    nexttile;
    plot(t,rolling(:,j),'LineWidth',1.1,'Tag',['rate_rms_' id]); time_axis(t,m);
    title([rate_names{j} '：过去 1 秒滚动均方根']); ylabel('角速度（rad/s）');
    nexttile;
    plot(t,100*pe(:,j),'LineWidth',1.1,'Tag',['late_position_error_' id]); time_axis(t,m);
    xlim([P.duration-3 P.duration]);
    title(sprintf('末 3 秒%s位置误差｜峰峰值 %.3f cm',axis_names{j},100*m.late_position_range_m(j)));
    ylabel('位置误差（cm）');
    sgtitle(sprintf('%s｜%s｜位置/速度：地面系；姿态：ZYX欧拉角；角速度：机体系',m.task_name,id),...
        'FontWeight','bold');
    save_figure(f,folder,['axis_' id]);
end
end

function values=euler_angles(q)
x=q(:,1); y=q(:,2); z=q(:,3); w=q(:,4);
values=[atan2d(2*(w.*x+y.*z),1-2*(x.^2+y.^2)),...
    asind(min(max(2*(w.*y-z.*x),-1),1)),atan2d(2*(w.*z+x.*y),1-2*(y.^2+z.^2))];
end

function f=make_figure(name,visible,position)
f=figure('Name',name,'Color','w','Position',position,'Visible',visible,...
    'DefaultAxesFontName','Microsoft YaHei','DefaultTextFontName','Microsoft YaHei');
end

function time_axis(t,m)
grid on; xlim([t(1) t(end)]); xlabel('Time (s)');
if m.task~=0 && isfinite(m.motion_finished_at_s)
    xline(m.motion_finished_at_s,':','Color',[.3 .3 .3],'HandleVisibility','off');
end
end

function save_figure(f,folder,name)
temporary_stem=tempname(folder);
temporary_png=[temporary_stem '.png']; temporary_fig=[temporary_stem '.fig'];
cleaner=onCleanup(@()remove_temporary(temporary_png,temporary_fig));
exportgraphics(f,temporary_png,'Resolution',160);
savefig(f,temporary_fig);
replace_output(temporary_png,fullfile(folder,[name '.png']));
replace_output(temporary_fig,fullfile(folder,[name '.fig']));
end

function replace_output(source,target)
for attempt=1:4
    [ok,message]=movefile(source,target,'f');
    if ok, return; end
    if attempt<4, pause(.25); end
end
error('hex:PlotReplaceFailed','Cannot replace %s: %s',target,message);
end

function remove_temporary(png_file,fig_file)
if isfile(png_file), delete(png_file); end
if isfile(fig_file), delete(fig_file); end
end
