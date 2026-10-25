function results=screen_hex_recovery(motors,factors,task,duration,observed,delay,baseline)
%SCREEN_HEX_RECOVERY Direct-source screening; writes failed trials honestly.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'blocks'),fullfile(root,'functions'));
if nargin<1, motors=1:6; end
if nargin<2, factors=[.8 .2 0]; end
if nargin<3, task=2; end
if nargin<4, duration=24; end
if nargin<5, observed=false; end
if nargin<6, delay=.03; end
if nargin<7, baseline=false; end
folder=fullfile(root,'output','recovery_tuning',char(datetime('now','Format','yyyyMMdd_HHmmss_SSS')));
mkdir(folder); fprintf('RECOVERY_SCREEN_FOLDER %s\n',folder);
results=struct([]); base=hex_recovery_parameters();
for motor=motors
    for factor=factors
        clear recovery_controller_step recovery_reference_step sensor_step controller_step
        P=base; P.fault_motor=motor; P.fault_factor=factor;
        P.task_family=task; P.duration=duration;
        P.observed_feedback=observed; P.noise_enabled=observed; P.wind_enabled=observed;
        P.feedback_delay=delay;
        P=hex_prepare_inputs(P); state=P.initial_state;
        groups=hex_recovery_block_parameters(P);
        times=(0:round(P.duration/P.base_dt))'*P.base_dt; N=numel(times);
        data=zeros(N,1+19+6+6+50+10); last=N; completed=true;
        failure='';
        for k=1:N
            t=times(k); [wind,loads]=wind_step(t,P);
            [feedback,~,~]=sensor_step(t,state,wind,loads,P);
            if baseline
                ref=task_reference(t,P); progress=[t;1;ref;norm(ref(1:3)-feedback(1:3));0];
            else
                [ref,progress]=recovery_reference_step(t,feedback,groups.task_reference);
            end
            try
                if baseline
                    [command,audit]=controller_step(feedback,ref,groups.controller_step);
                    audit=[audit;ones(6,1);0];
                else
                    [command,audit]=recovery_controller_step(feedback,ref,groups.controller_step);
                end
            catch problem
                failure=problem.message; completed=false; last=max(k-1,1); break
            end
            data(k,:)=[t;state;ref;command;audit;progress]';
            if envelope_step(state,P), completed=false; last=k; break; end
            if k<N
                [segments,~]=fault_schedule(t,P);
                state=plant_step(state,command,segments,wind,loads,P);
            end
        end
        data=data(1:last,:); s=data(:,2:20); ref=data(:,21:26);
        errors=vecnorm(s(:,1:3)-ref(:,1:3),2,2);
        tilt=acosd(min(max(1-2*(s(:,7).^2+s(:,8).^2),-1),1));
        post=data(:,1)>=P.fault_start; late=data(:,1)>=max(P.fault_start,P.duration-3);
        item=struct('motor',motor,'factor',factor,'task',task,'observed',observed,...
            'baseline',baseline,...
            'duration_s',data(end,1),'completed',completed,'failure',failure,...
            'feedback_delay_s',P.feedback_delay,...
            'rmse_m',sqrt(mean(errors.^2)),'peak_error_m',max(errors),...
            'late_rmse_m',sqrt(mean(errors(late).^2)),'final_error_m',errors(end),...
            'peak_tilt_deg',max(tilt),'post_peak_tilt_deg',max(tilt(post)),...
            'peak_rate_rad_s',max(vecnorm(s(:,11:13),2,2)),...
            'minimum_altitude_m',min(s(:,3)),...
            'phase_s',data(end,83),'estimated_eta',data(end,76:81));
        synthetic=struct('parameters',P,'time',data(:,1),'state',s,'reference',ref,...
            'motor_command',data(:,27:32),'control_audit',data(:,33:82),...
            'recovery_reference_audit',data(:,83:92));
        item.acceptance=hex_recovery_metrics(synthetic);
        if isempty(results), results=item; else, results(end+1)=item; end %#ok<AGROW>
        name=sprintf('task%d_m%d_eta%02d_observed%d',task,motor,round(100*factor),observed);
        save(fullfile(folder,[name '.mat']),'P','data','item');
        fprintf('RECOVERY %s complete=%d duration=%.2f late=%.4f peak=%.4f tilt=%.2f phase=%.2f eta=%s\n',...
            name,completed,item.duration_s,item.late_rmse_m,item.peak_error_m,item.peak_tilt_deg,item.phase_s,mat2str(item.estimated_eta,3));
        fid=fopen(fullfile(folder,'screen_latest.json'),'w'); fprintf(fid,'%s',jsonencode(results,PrettyPrint=true)); fclose(fid);
    end
end
end
