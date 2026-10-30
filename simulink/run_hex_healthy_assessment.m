function report=run_hex_healthy_assessment(options)
%RUN_HEX_HEALTHY_ASSESSMENT Four healthy tasks using the current recovery model.
% Run in the editor or enter: report=run_hex_healthy_assessment;
% The original model, parameters and my_simulation.m are not saved or edited.
arguments
    options.Duration (1,1) double {mustBeGreaterThanOrEqual(options.Duration,10)}=24
    options.Noise (1,1) logical=true
    options.Wind (1,1) logical=true
    options.ObservedFeedback (1,1) logical=true
    options.Delay (1,1) double {mustBeNonnegative}=.03
    options.Seed (1,1) double=20261006
    options.OutputName (1,1) string=""
    options.ShowFigures (1,1) logical=true
    options.ReuseSavedResults (1,1) logical=false
end
root=fileparts(mfilename('fullpath'));
addpath(root);
P=hex_recovery_parameters();
P.duration=options.Duration;
P.fault_motor=0; P.fault_factor=1;
P.motion_start=1; P.motion_end=7;
P.heading=0; P.displacement=1.2; P.radius=.3; P.turn=2*pi;
P.observed_feedback=options.ObservedFeedback;
P.noise_enabled=options.Noise; P.wind_enabled=options.Wind;
P.feedback_delay=options.Delay; P.seed=options.Seed;
P=hex_prepare_inputs(P);
if strlength(options.OutputName)==0
    options.OutputName="healthy_four_tasks_"+string(datetime('now','Format','yyyyMMdd_HHmmss'));
end
folder=fullfile(root,'output','recovery',P.control_tuning_version,char(options.OutputName));
if ~isfolder(folder), mkdir(folder); end
task_names={'悬停','直线','绕圈','升降'};
case_names={'00_hover','01_line','02_circle','03_vertical'};
items=struct([]);
for k=1:4
    P.task_family=k-1;
    fprintf('HEALTHY_TASK %d/4 %s | wind=%d noise=%d delay=%.3f s\n',...
        k,case_names{k},P.wind_enabled,P.noise_enabled,P.feedback_delay);
    case_folder=fullfile(folder,case_names{k});
    saved_result=fullfile(case_folder,'result.mat');
    if options.ReuseSavedResults && isfile(saved_result)
        cached=load(saved_result,'result');
        assert(isequaln(cached.result.parameters,P),'Saved results use different parameters.');
        r=cached.result;
        fprintf('HEALTHY_REUSE %s | saved parameters match\n',case_names{k});
    else
        r=run_hex_recovery(Parameters=P,SaveResults=true,...
            OutputName=fullfile(options.OutputName,case_names{k}));
    end
    assert(strcmp(r.termination,'completed'),'A task stopped early; inspect the saved result.');
    assert(all(isfinite(r.state),'all') && all(isfinite(r.control_audit),'all'),...
        'Nonfinite state or controller data.');
    assert(all(abs(r.effectiveness-1)<1e-12,'all'),'The plant was not fully healthy.');
    assert(size(r.state,1)==round(P.duration/P.export_dt)+1,'Unexpected sample count.');
    [item,motor_summary]=measure(r,task_names{k},case_names{k});
    if isempty(items), items=item; else, items(k)=item; end
    writetable(motor_summary,fullfile(case_folder,'motor_metrics.csv'),'Encoding','UTF-8');
    fid=fopen(fullfile(case_folder,'metrics.json'),'w','n','UTF-8');
    assert(fid>=0,'Cannot open metrics output.');
    fprintf(fid,'%s',jsonencode(item,PrettyPrint=true)); fclose(fid);
    figures=plot_hex_healthy_assessment(r,case_folder,item,options.ShowFigures);
    if ~options.ShowFigures, close(figures); end
    fprintf('HEALTHY_RESULT %s position_rmse=%.6f m peak=%.6f m late_rate_rms=%.4f rad/s saturated=%.3f%%\n',...
        case_names{k},item.position_rmse_m,item.position_peak_m,item.late_body_rate_rms_rad_s,...
        item.any_motor_saturation_percent);
end
report=struct('model','hex_recovery','plot_layout','separate_xyz',...
    'control_tuning_version',P.control_tuning_version,...
    'created',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')),...
    'output_folder',folder,'settings',options,'cases',items);
save(fullfile(folder,'assessment.mat'),'report','-v7');
fid=fopen(fullfile(folder,'assessment.json'),'w','n','UTF-8');
assert(fid>=0,'Cannot open assessment output.');
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
writetable(struct2table(items),fullfile(folder,'metrics.csv'),'Encoding','UTF-8');
summary_figures=plot_summary(items,options);
summary_names={'four_tasks_comparison','four_tasks_x_comparison',...
    'four_tasks_y_comparison','four_tasks_z_comparison'};
for j=1:numel(summary_figures)
    exportgraphics(summary_figures(j),fullfile(folder,[summary_names{j} '.png']),'Resolution',170);
    savefig(summary_figures(j),fullfile(folder,[summary_names{j} '.fig']));
end
if ~options.ShowFigures, close(summary_figures); end
write_report(report);
fprintf('HEALTHY_ASSESSMENT_SAVED %s\n',folder);
end

function [item,motor_summary]=measure(r,task_name,case_name)
t=r.time; P=r.parameters; s=r.state;
pe=s(:,1:3)-r.reference(:,1:3); ve=s(:,4:6)-r.reference(:,4:6);
pn=vecnorm(pe,2,2); vn=vecnorm(ve,2,2); w=s(:,11:13);
late=t>=P.duration-3;
tilt=acosd(min(max(1-2*(s(:,7).^2+s(:,8).^2),-1),1));
% Duration is integrated over logged intervals, excluding the final endpoint.
tol=max(1e-8,1e-6*(P.omega_max-P.omega_min));
upper=r.motor_command>=P.omega_max-tol;
lower=r.motor_command<=P.omega_min+tol;
sat=upper|lower; dt=diff(t); duration=t(end)-t(1);
motor_sat_seconds=sum(double(sat(1:end-1,:)).*dt,1);
any_sat_seconds=sum(double(any(sat(1:end-1,:),2)).*dt);
command_bounds=all(r.motor_command>=P.omega_min-1e-8 & r.motor_command<=P.omega_max+1e-8,'all');
integral_bounds=all(abs(r.control_audit(:,13:15))<=P.control.velocity_integral_limit_m_s2'+1e-8,'all') && ...
    all(abs(r.control_audit(:,16:18))<=P.control.rate_integral_limit_rad_s2'+1e-8,'all');
% The governor can delay the end of motion. Hover has no motion interval.
if P.task_family==0
    end_time=0;
else
    reached=find(r.recovery_reference_audit(:,1)>=P.motion_end-1e-9,1);
    end_time=NaN; if ~isempty(reached), end_time=t(reached); end
end
band=.05; minimum_hold=2; settle_wait=NaN;
if isfinite(end_time)
    last_outside=find(t>=end_time & pn>band,1,'last');
    first_inside=find(t>=end_time,1);
    if ~isempty(last_outside), first_inside=last_outside+1; end
    if first_inside<=numel(t) && t(end)-t(first_inside)>=minimum_hold-1e-9
        settle_wait=t(first_inside)-end_time;
    end
end
item=struct('task',P.task_family,'task_name',task_name,'case_name',case_name,...
    'duration_s',t(end),'position_rmse_m',sqrt(mean(pn.^2)),...
    'position_axis_rmse_m',sqrt(mean(pe.^2,1)),...
    'position_peak_m',max(pn),'position_final_m',pn(end),...
    'velocity_rmse_m_s',sqrt(mean(vn.^2)),...
    'velocity_axis_rmse_m_s',sqrt(mean(ve.^2,1)),...
    'velocity_peak_error_m_s',max(vn),'maximum_tilt_deg',max(tilt),...
    'maximum_body_rate_rad_s',max(vecnorm(w,2,2)),...
    'maximum_rp_rate_rad_s',max(vecnorm(w(:,1:2),2,2)),...
    'body_rate_axis_peak_rad_s',max(abs(w),[],1),...
    'late_position_rmse_m',sqrt(mean(pn(late).^2)),...
    'late_velocity_rmse_m_s',sqrt(mean(vn(late).^2)),...
    'late_position_range_m',max(pe(late,:),[],1)-min(pe(late,:),[],1),...
    'late_body_rate_rms_rad_s',sqrt(mean(sum(w(late,:).^2,2))),...
    'late_body_rate_axis_rms_rad_s',sqrt(mean(w(late,:).^2,1)),...
    'motion_finished_at_s',end_time,'position_band_m',band,...
    'minimum_hold_s',minimum_hold,'position_band_wait_s',settle_wait,...
    'any_motor_saturation_s',any_sat_seconds,...
    'any_motor_saturation_percent',100*any_sat_seconds/duration,...
    'saturation_tolerance_rad_s',tol,...
    'maximum_command_rad_s',max(r.motor_command,[],'all'),...
    'minimum_command_rad_s',min(r.motor_command,[],'all'),...
    'commands_bounded',command_bounds,'integrals_bounded',integral_bounds,...
    'mission_phase_complete',r.recovery_reference_audit(end,1)>=P.motion_end,...
    'late_window_after_motion',isfinite(end_time) && end_time<=P.duration-3,...
    'quaternion_norm_error',max(abs(vecnorm(s(:,7:10),2,2)-1)));
motor_summary=table((1:6)',min(r.motor_command,[],1)',max(r.motor_command,[],1)',...
    max(s(:,14:19),[],1)',motor_sat_seconds',100*motor_sat_seconds'/duration,...
    'VariableNames',{'motor','minimum_command_rad_s','maximum_command_rad_s',...
    'maximum_actual_speed_rad_s','saturation_seconds','saturation_percent'});
assert(command_bounds && integral_bounds,'Command or integral bounds were violated.');
assert(item.quaternion_norm_error<1e-10,'Quaternion normalization failed.');
end

function figures=plot_summary(items,options)
visible='off'; if options.ShowFigures, visible='on'; end
figures=gobjects(1,4);
f=figure('Name','健康四任务指标对比','Color','w','Position',[60 60 1500 880],...
    'Visible',visible,'DefaultAxesFontName','Microsoft YaHei','DefaultTextFontName','Microsoft YaHei');
figures(1)=f;
tiledlayout(f,2,3,'TileSpacing','compact','Padding','loose');
names={items.task_name};
nexttile; bar(100*[[items.position_rmse_m]',[items.position_peak_m]',[items.position_final_m]']);
xticks(1:4); xticklabels(names); ylabel('位置误差（cm）'); title('整段位置跟踪精度');
legend('均方根','最大值','最终值','Location','northwest'); grid on;
nexttile; bar([items.velocity_rmse_m_s]); xticks(1:4); xticklabels(names);
ylabel('速度误差（m/s）'); title('整段三维速度误差均方根'); grid on;
nexttile; bar([items.maximum_tilt_deg]); xticks(1:4); xticklabels(names);
ylabel('倾角（°）'); title('最大实际倾角'); grid on;
nexttile; bar([items.late_body_rate_rms_rad_s]);
xticks(1:4); xticklabels(names); ylabel('角速度（rad/s）');
title('末 3 秒三维真实角速度均方根'); grid on;
nexttile; bar(100*[items.late_position_rmse_m]); xticks(1:4); xticklabels(names);
ylabel('位置误差（cm）'); title('末 3 秒三维位置误差均方根'); grid on;
nexttile; bar([items.any_motor_saturation_percent]); xticks(1:4); xticklabels(names);
ylabel('时间占比（%）'); ylim([0 max(1,1.2*max([items.any_motor_saturation_percent]))]);
title('至少一台电机指令触及上下限'); grid on;
for k=1:4
    text(k,items(k).any_motor_saturation_percent+.03,sprintf('%.3g%%',items(k).any_motor_saturation_percent),...
        'HorizontalAlignment','center');
end
sgtitle(sprintf('健康四任务对比｜每任务 %.0f s｜风=%d，噪声=%d，观测反馈=%d｜延迟 %.0f ms',...
    options.Duration,options.Wind,options.Noise,options.ObservedFeedback,1000*options.Delay),'FontWeight','bold');
position_axis=vertcat(items.position_axis_rmse_m);
velocity_axis=vertcat(items.velocity_axis_rmse_m_s);
rate_peak=vertcat(items.body_rate_axis_peak_rad_s);
rate_late=vertcat(items.late_body_rate_axis_rms_rad_s);
ids={'x','y','z'}; rate_names={'滚转 p','俯仰 q','偏航 r'};
for j=1:3
    f=figure('Name',['四任务 ' ids{j} '指标'],'Color','w','Position',[80 70 1200 820],...
        'Visible',visible,'DefaultAxesFontName','Microsoft YaHei','DefaultTextFontName','Microsoft YaHei');
    figures(j+1)=f; tiledlayout(f,2,2,'TileSpacing','compact','Padding','loose');
    nexttile; bar(100*position_axis(:,j)); ylabel('位置误差（cm）');
    title([ids{j} '位置误差均方根（地面系）']); xticks(1:4); xticklabels(names); grid on;
    nexttile; bar(velocity_axis(:,j)); ylabel('速度误差（m/s）');
    title([ids{j} '速度误差均方根（地面系）']); xticks(1:4); xticklabels(names); grid on;
    nexttile; bar(rate_peak(:,j)); ylabel('角速度（rad/s）');
    title([rate_names{j} '：真实机体角速度绝对峰值']); xticks(1:4); xticklabels(names); grid on;
    nexttile; bar(rate_late(:,j)); ylabel('角速度（rad/s）');
    title([rate_names{j} '：末 3 秒真实机体角速度均方根']); xticks(1:4); xticklabels(names); grid on;
    sgtitle(sprintf('健康四任务｜%s独立对比｜每任务 %.0f s',ids{j},options.Duration),'FontWeight','bold');
end
end

function write_report(report)
fid=fopen(fullfile(report.output_folder,'README.md'),'w','n','UTF-8');
assert(fid>=0,'Cannot open report output.'); cleaner=onCleanup(@()fclose(fid));
o=report.settings;
fprintf(fid,'# 四个健康任务的跟踪与稳定性观察\n\n');
fprintf(fid,'模型：`%s`；控制参数：`%s`；每任务 %.0f 秒，电机健康。\n\n',...
    report.model,report.control_tuning_version,o.Duration);
fprintf(fid,'风=%d，测量噪声=%d，观测反馈=%d，反馈延迟=%.0f ms，随机种子=%.0f。四任务共享同一外部风/噪声序列。\n\n',...
    o.Wind,o.Noise,o.ObservedFeedback,1000*o.Delay,o.Seed);
fprintf(fid,'轨迹：初始位置 [0,0,2.5] m；直线向地面系 x 正方向移动 1.2 m；绕圈半径 0.3 m、一整圈；升降向上 1.2 m。名义运动时间 1–7 s，进度调节器可以推迟实际结束时刻。\n\n');
fprintf(fid,'每个子目录：`tracking.png/.fig`（标量误差和三维轨迹总览）、`stability.png/.fig`（总倾角与执行器余量）、`axis_x.png/.fig`、`axis_y.png/.fig`、`axis_z.png/.fig`（三个方向独立绘制），以及原始 `result.mat/signals.csv`、`metrics.json` 和 `motor_metrics.csv`。总目录有四任务总对比图、三个方向各自的对比图与指标 CSV/JSON。共 24 张评估图，各存 PNG/FIG。\n\n');
fprintf(fid,'方向图：位置与速度为地面系 x/y/z 分量；姿态为 ZYX 欧拉角（滚转、俯仰、偏航），角速度为机体系 FLU 分量，两者的坐标系在图中注明。三个方向不叠加在同一张时序图中。\n\n');
fprintf(fid,'|任务|位置 RMSE/cm|最大位置误差/cm|最终位置误差/cm|速度 RMSE/(m/s)|最大倾角/°|末3秒角速度 RMS/(rad/s)|电机饱和占比|实际运动结束/s|\n');
fprintf(fid,'|---|---:|---:|---:|---:|---:|---:|---:|---:|\n');
for k=1:numel(report.cases)
    m=report.cases(k);
    fprintf(fid,'|%s|%.3f|%.3f|%.3f|%.4f|%.2f|%.4f|%.3f%%|%.2f|\n',...
        m.task_name,100*m.position_rmse_m,100*m.position_peak_m,100*m.position_final_m,...
        m.velocity_rmse_m_s,m.maximum_tilt_deg,m.late_body_rate_rms_rad_s,...
        m.any_motor_saturation_percent,m.motion_finished_at_s);
end
fprintf(fid,'\n统计口径：位置/速度三维 RMSE 为 sqrt(mean(sum(error.^2,2)))，参考为模型实际输出的受控参考。角速度来自 state(:,11:13) 的机体系真值，未掺入陀螺仪测量噪声。末段统计使用最后 3 秒；同时报告该窗口是否已在运动结束之后。\n\n');
fprintf(fid,'饱和采用电机转速指令触及物理上下限的判据，容差为全转速范围的 1e-6，按相邻样本间隔计时。未饱和不代表执行器响应没有滞后。\n\n');
fprintf(fid,'位置带等待时间：从实际运动结束起，找到三维位置误差进入 5 cm 带并一直保持到仿真末尾的最早时刻，要求剩余保持至少 2 秒；未观察到则为 NaN（JSON 中为 null）。该带仅供描述，不是稳定性验收标准。持续随机风下不要求误差趋于零。\n\n');
fprintf(fid,'本批次没有施加独立的脉冲/阶跃外扰，因而没有报告“外扰恢复时间”或“阶跃超调量”。图中的滚动角速度 RMS 为过去 1 秒窗口，起始阶段使用已有样本。只绘制和统计本次有限时长结果，不把四次仿真称为稳定性证明。\n');
end
