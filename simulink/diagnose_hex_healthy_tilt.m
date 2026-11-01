function report=diagnose_hex_healthy_tilt(assessment_folder,options)
%DIAGNOSE_HEX_HEALTHY_TILT Controlled hover comparisons with unchanged gains.
arguments
    assessment_folder (1,1) string
    options.ShowFigures (1,1) logical=true
    options.ReuseSavedResults (1,1) logical=true
end
root=fileparts(mfilename('fullpath')); addpath(root);
saved=load(fullfile(assessment_folder,'00_hover','result.mat'),'result');
base=saved.result; P=base.parameters;
assert(P.task_family==0 && P.fault_motor==0,'A saved healthy hover case is required.');
assert(P.wind_enabled && P.noise_enabled && P.observed_feedback && abs(P.feedback_delay-.03)<1e-10,...
    'The reference comparison requires wind/noise/observed feedback with 30-ms delay.');
names={'full','without_pv_noise','wind_only','zero_delay','ideal'};
labels={'完整设置','去除位置与速度测量噪声','关闭全部测量噪声，保留风','完整噪声与风，反馈延迟为零','无风无测量噪声，保留延迟'};
folder=fullfile(assessment_folder,'tilt_diagnostic'); if ~isfolder(folder), mkdir(folder); end
traces=cell(1,5); traces{1}=base; items=struct([]);
for k=1:5
    Q=P;
    if k==2
        Q.exogenous(:,10:15)=0;
    elseif k==3
        Q.noise_enabled=false; Q=hex_prepare_inputs(Q);
    elseif k==4
        Q.feedback_delay=0;
    elseif k==5
        Q.noise_enabled=false; Q.wind_enabled=false; Q=hex_prepare_inputs(Q);
    end
    if k>1
        case_folder=fullfile(folder,names{k});
        if ~isfolder(case_folder), mkdir(case_folder); end
        saved_file=fullfile(case_folder,'result.mat');
        if options.ReuseSavedResults && isfile(saved_file)
            cached=load(saved_file,'result');
            assert(isequaln(cached.result.parameters,Q),'Diagnostic cached parameters differ.');
            result=cached.result;
        else
            fprintf('TILT_DIAGNOSTIC %d/4 %s\n',k-1,names{k});
            result=run_hex_simulink(Parameters=Q,SaveResults=false);
            save(saved_file,'result','-v7');
        end
        traces{k}=result;
    end
    r=traces{k}; s=r.state; qd=r.control_audit(:,28:31);
    assert(strcmp(r.termination,'completed') && all(isfinite(s),'all'));
    actual=tilt(s(:,7:10)); desired=tilt(qd);
    ea=euler_angles(s(:,7:10)); ed=euler_angles(qd);
    late=r.time>=Q.duration-3;
    item=struct('case_name',names{k},'description',labels{k},...
        'actual_tilt_peak_deg',max(actual),'desired_tilt_peak_deg',max(desired),...
        'actual_tilt_mean_deg',mean(actual),'desired_tilt_mean_deg',mean(desired),...
        'actual_euler_step_rms_deg',sqrt(mean(diff(ea,1,1).^2,1)),...
        'desired_euler_step_rms_deg',sqrt(mean(diff(ed,1,1).^2,1)),...
        'body_rate_rms_rad_s',sqrt(mean(sum(s(:,11:13).^2,2))),...
        'late_body_rate_rms_rad_s',sqrt(mean(sum(s(late,11:13).^2,2))),...
        'position_rmse_m',sqrt(mean(sum((s(:,1:3)-r.reference(:,1:3)).^2,2))),...
        'maximum_controller_damage_estimate',max(r.control_audit(:,50)));
    if isempty(items), items=item; else, items(k)=item; end
    fprintf('TILT_RESULT %s actual_peak=%.4f desired_peak=%.4f late_rate_rms=%.5f\n',...
        names{k},item.actual_tilt_peak_deg,item.desired_tilt_peak_deg,item.late_body_rate_rms_rad_s);
end
report=struct('source',fullfile(assessment_folder,'00_hover','result.mat'),...
    'description','Healthy hover; same gains/seed; changing only the stated input channels or delay.',...
    'cases',items);
fid=fopen(fullfile(folder,'diagnosis.json'),'w','n','UTF-8'); assert(fid>=0);
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
writetable(struct2table(items),fullfile(folder,'diagnosis.csv'),'Encoding','UTF-8');
visible='off'; if options.ShowFigures, visible='on'; end
f=figure('Name','悬停倾角来源对照','Color','w','Position',[70 55 1450 950],...
    'Visible',visible,'DefaultAxesFontName','Microsoft YaHei','DefaultTextFontName','Microsoft YaHei');
tiledlayout(f,3,2,'TileSpacing','compact','Padding','loose');
for k=1:5
    nexttile; plot(traces{k}.time,tilt(traces{k}.state(:,7:10)),'LineWidth',1.1); hold on;
    plot(traces{k}.time,tilt(traces{k}.control_audit(:,28:31)),'--','LineWidth',.8);
    grid on; xlabel('Time (s)'); ylabel('倾角（°）'); title(labels{k});
    legend('实际倾角真值','期望倾角指令','Location','best');
end
nexttile; bar([items.late_body_rate_rms_rad_s]); grid on;
ylim([0 max(.01,1.15*max([items.late_body_rate_rms_rad_s]))]);
xticks(1:5); xticklabels({'完整','无位置速度噪声','仅风','零延迟','理想'});
xtickangle(15); ylabel('三维角速度均方根（rad/s）'); title('末 3 秒真实角速度对比');
sgtitle('健康悬停｜同一增益与随机种子｜各 24 秒｜闭环对照');
exportgraphics(f,fullfile(folder,'tilt_sources.png'),'Resolution',160);
savefig(f,fullfile(folder,'tilt_sources.fig'));
if ~options.ShowFigures, close(f); end
fprintf('TILT_DIAGNOSTIC_SAVED %s\n',folder);
end

function values=tilt(q)
values=acosd(min(max(1-2*(q(:,1).^2+q(:,2).^2),-1),1));
end

function values=euler_angles(q)
x=q(:,1); y=q(:,2); z=q(:,3); w=q(:,4);
values=[atan2d(2*(w.*x+y.*z),1-2*(x.^2+y.^2)),...
    asind(min(max(2*(w.*y-z.*x),-1),1)),...
    atan2d(2*(w.*z+x.*y),1-2*(y.^2+z.^2))];
end
