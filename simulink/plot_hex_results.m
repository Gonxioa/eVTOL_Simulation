function [figure_handle,detail_figures]=plot_hex_results(r,folder,motor_indices)
%PLOT_HEX_RESULTS Overview, individual motor responses and gyro/truth comparison.
% Example: plot_hex_results(r,'',3) displays motor 3 without saving figures.
% The default 1:6 creates one independent response figure for each motor.
if nargin<2, folder=''; end
if nargin<3, motor_indices=1:6; end
validateattributes(motor_indices,{'numeric'},{'vector','nonempty','integer','>=',1,'<=',6},mfilename,'motor_indices');
motor_indices=unique(motor_indices(:)','stable');
t=r.time; s=r.state; ref=r.reference;
% The overview shows the first selected motor; detailed figures show them all.
overview_motor=motor_indices(1);
actual_speed=s(:,13+overview_motor); command_speed=r.motor_command(:,overview_motor);
% sensor_step uses install'*w for column vectors; logs contain row vectors.
true_rates=s(:,11:13)*r.parameters.sensor.installation_matrix;
gyro_rates=r.imu(:,4:6);
rate_colors=[0 .447 .741;.85 .325 .098;.494 .184 .556];
axis_names={'x','y','z'};
plot_font='Microsoft YaHei';
figure_handle=figure('Name','六旋翼仿真结果','Color','w','Position',[80 80 1400 850], ...
    'DefaultAxesFontName',plot_font,'DefaultTextFontName',plot_font);
tiledlayout(3,3,'TileSpacing','compact','Padding','loose');
nexttile; plot3(s(:,1),s(:,2),s(:,3),'LineWidth',1.3); hold on
plot3(ref(:,1),ref(:,2),ref(:,3),'--','LineWidth',1); grid on; axis equal
xlabel('（米）');
ylabel('（米）','Units','normalized','Position',[.02 .02 0], ...
    'HorizontalAlignment','left','Rotation',0);
zlabel('（米）');
title('三维飞行轨迹'); legend('实际轨迹','期望轨迹','FontName',plot_font);
nexttile; plot(t,s(:,1:3)-ref(:,1:3)); grid on; title('位置跟踪误差'); ylabel('位置误差（米）');
legend('x','y','z','FontName',plot_font);
nexttile; plot(t,s(:,4:6)); grid on; title('速度'); ylabel('速度（m/s）');
legend('x','y','z','FontName',plot_font);
nexttile;
plot(t,actual_speed,'LineWidth',1.3,'DisplayName','实际转速'); hold on
plot(t,command_speed,'--','LineWidth',1,'DisplayName','指令转速');
grid on; title(sprintf('%d号电机转速与指令',overview_motor)); ylabel('转速（rad/s）');
legend('Location','best','FontName',plot_font);
nexttile; plot(t,r.imu(:,1:3)); grid on; title('加速度计输出'); ylabel('比力（m/s²）');
legend('x轴','y轴','z轴','FontName',plot_font);
nexttile; hold on
for axis_index=1:3
    plot(t,true_rates(:,axis_index),'Color',rate_colors(axis_index,:),'LineWidth',1.2, ...
        'DisplayName',[axis_names{axis_index} '轴真值']);
    plot(t,gyro_rates(:,axis_index),'--','Color',rate_colors(axis_index,:),'LineWidth',.8, ...
        'DisplayName',[axis_names{axis_index} '轴测量']);
end
grid on; title('陀螺仪与真值'); ylabel('角速度（rad/s）');
legend('Location','best','NumColumns',3,'FontSize',8,'FontName',plot_font);
nexttile; plot(t,r.control_audit(:,13:15)); grid on; title('速度环积分项'); ylabel('补偿加速度（m/s²）'); xlabel('Time (s)');
legend('东向','北向','竖直方向','FontName',plot_font);
nexttile; plot(t,r.control_audit(:,16:18)); grid on; title('角速度环积分项'); ylabel('角加速度（rad/s²）'); xlabel('Time (s)');
legend('横滚','俯仰','偏航','FontName',plot_font);
nexttile; stairs(t,r.effectiveness); ylim([-.05 1.05]); grid on; title('电机效能'); ylabel('失效系数'); xlabel('Time (s)');
legend('1','2','3','4','5','6', ...
    'Location','best','NumColumns',3,'FontSize',8,'FontName',plot_font);
if r.parameters.fault_motor==0
    fault_label='全部电机健康';
else
    fault_label=sprintf('故障电机 %d｜剩余效能 %.3g%%',r.parameters.fault_motor,100*r.parameters.fault_factor);
end
if strcmp(r.termination,'completed')
    termination_label='仿真完成';
elseif strcmp(r.termination,'physical_envelope_exit')
    termination_label='状态越界，提前停止';
else
    termination_label=r.termination;
end
sgtitle(sprintf('六旋翼仿真｜任务 %d｜%s｜%s',r.parameters.task_family,fault_label,termination_label), ...
    'Interpreter','none','FontName',plot_font);
if ~isempty(folder)
    if ~isfolder(folder), mkdir(folder); end
    exportgraphics(figure_handle,fullfile(folder,'results.png'),'Resolution',140);
    savefig(figure_handle,fullfile(folder,'results.fig'));
end

detail_figures.motor=gobjects(1,numel(motor_indices));
for motor_position=1:numel(motor_indices)
    motor_index=motor_indices(motor_position);
    actual_speed=s(:,13+motor_index); command_speed=r.motor_command(:,motor_index);
    motor_figure=figure('Name',sprintf('%d号电机响应',motor_index), ...
        'Color','w','Position',[100 100 1100 720], ...
        'DefaultAxesFontName',plot_font,'DefaultTextFontName',plot_font);
    detail_figures.motor(motor_position)=motor_figure;
    tiledlayout(2,1,'TileSpacing','compact','Padding','loose');
    motor_axes=gobjects(2,1);
    motor_axes(1)=nexttile;
    plot(t,actual_speed,'LineWidth',1.4,'DisplayName','实际转速'); hold on
    plot(t,command_speed,'--','LineWidth',1,'DisplayName','转速指令');
    grid on; title('实际转速与转速指令'); ylabel('转速（rad/s）');
    legend('Location','best','FontName',plot_font);
    motor_axes(2)=nexttile;
    plot(t,actual_speed-command_speed,'LineWidth',1.1); hold on
    yline(0,':','HandleVisibility','off'); grid on
    title('转速跟踪误差：实际转速－指令转速'); ylabel('转速误差（rad/s）'); xlabel('Time (s)');
    linkaxes(motor_axes,'x');
    sgtitle(sprintf('%d号电机响应｜任务 %d',motor_index,r.parameters.task_family), ...
        'Interpreter','none','FontName',plot_font);
    if ~isempty(folder)
        file_stem=sprintf('motor_%d_response',motor_index);
        exportgraphics(motor_figure,fullfile(folder,[file_stem '.png']),'Resolution',140);
        savefig(motor_figure,fullfile(folder,[file_stem '.fig']));
    end
end

detail_figures.gyroscope=figure('Name','陀螺仪与真实角速度对比', ...
    'Color','w','Position',[120 120 1300 850], ...
    'DefaultAxesFontName',plot_font,'DefaultTextFontName',plot_font);
tiledlayout(3,2,'TileSpacing','compact','Padding','loose');
gyro_axes=gobjects(6,1);
for axis_index=1:3
    label=upper(axis_names{axis_index});
    gyro_axes(2*axis_index-1)=nexttile;
    plot(t,true_rates(:,axis_index),'LineWidth',1.3,'DisplayName','真实角速度', ...
        'Tag',['gyro_true_' axis_names{axis_index}]); hold on
    plot(t,gyro_rates(:,axis_index),'--','LineWidth',.9,'DisplayName','陀螺仪测量值', ...
        'Tag',['gyro_measured_' axis_names{axis_index}]);
    grid on; title([label ' 轴：真实角速度与测量值']);
    ylabel('角速度（rad/s）','Units','normalized','Position',[-.09 .5 0]);
    legend('Location','best','FontName',plot_font);
    gyro_axes(2*axis_index)=nexttile;
    gyro_error=gyro_rates(:,axis_index)-true_rates(:,axis_index);
    plot(t,gyro_error,'Color',[.75 .15 .15],'LineWidth',.9, ...
        'Tag',['gyro_error_' axis_names{axis_index}]); grid on
    title(sprintf('%s 轴误差：测量值－真值｜均方根 %.4g rad/s',label,sqrt(mean(gyro_error.^2))));
    ylabel('角速度误差（rad/s）','Units','normalized','Position',[-.09 .5 0]);
    if axis_index==3
        xlabel(gyro_axes(5),'Time (s)'); xlabel(gyro_axes(6),'Time (s)');
    end
end
linkaxes(gyro_axes,'x');
sgtitle(sprintf('陀螺仪输出与真实角速度对比｜任务 %d',r.parameters.task_family), ...
    'Interpreter','none','FontName',plot_font);
if ~isempty(folder)
    exportgraphics(detail_figures.gyroscope,fullfile(folder,'gyroscope_vs_truth.png'),'Resolution',140);
    savefig(detail_figures.gyroscope,fullfile(folder,'gyroscope_vs_truth.fig'));
end
end
