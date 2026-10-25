function results=check_hex_baseline_candidates()
%CHECK_HEX_BASELINE_CANDIDATES Verify gain candidates in an isolated SLX copy.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'blocks'));
folder=fullfile(root,'output','baseline_tuning');
work=fullfile(folder,'candidate_model'); if ~isfolder(work), mkdir(work); end
mdl='hex_evtol'; copyfile(fullfile(root,[mdl '.slx']),fullfile(work,[mdl '.slx']));
load_system(fullfile(work,[mdl '.slx']));
cleanup=onCleanup(@()close_model(mdl));
base=hex_parameters(); base.exogenous=zeros(1202,24);
apply(base); set_param(mdl,'FastRestart','off');
gains=[4 10 16 40 .5;5 12 18 45 .6];
results=struct([]);
for candidate=1:size(gains,1)
    for scenario=1:7
        P=base; g=gains(candidate,:);
        P.control.position_p(1:2)=g(1); P.control.velocity_p(1:2)=g(2);
        P.control.attitude_p(1:2)=g(3); P.control.rate_p(1:2)=g(4); P.control.rate_d(1:2)=g(5);
        P.task_family=2;
        names={'circle','line','vertical','circle_large','circle_observed','circle_noise_wind','circle_delay_30ms'};
        if scenario==2, P.task_family=1; P.heading=pi/4; end
        if scenario==3, P.task_family=3; end
        if scenario==4, P.radius=.5; P.heading=pi/3; end
        if scenario>=5, P.observed_feedback=true; end
        if scenario>=6, P.noise_enabled=true; P.wind_enabled=true; end
        if scenario==7, P.feedback_delay=.03; end
        P=hex_prepare_inputs(P); P.exogenous(end+1:1202,:)=0;
        input=Simulink.SimulationInput(mdl);
        input=input.setVariable('P',P,'Workspace',mdl);
        groups=hex_block_parameters(P); fields=fieldnames(groups);
        for j=1:numel(fields)
            input=input.setVariable(['P_' fields{j}],groups.(fields{j}),'Workspace',mdl);
        end
        out=sim(input); s=values(out.state_log); ref=values(out.reference_log); cmd=values(out.motor_command_log);
        sample=min(401,numel(out.reference_log.Time));
        assert(norm(ref(sample,:)'-task_reference(out.reference_log.Time(sample),P))<1e-10,...
            'The SLX did not apply the requested trajectory parameters.');
        actual_wind=values(out.wind_log);
        assert(P.wind_enabled==any(abs(actual_wind)>1e-12,'all'),...
            'The SLX did not apply the requested wind switch.');
        errors=vecnorm(s(:,1:3)-ref(:,1:3),2,2);
        tilt=acosd(min(max(1-2*(s(:,7).^2+s(:,8).^2),-1),1));
        item=struct('candidate',candidate,'name',names{scenario},'gains',g,'control',P.control,...
            'rmse_m',sqrt(mean(errors.^2)),'maximum_error_m',max(errors),'final_error_m',errors(end),...
            'maximum_tilt_deg',max(tilt),'maximum_command_rad_s',max(cmd,[],'all'),...
            'actual_duration_s',out.state_log.Time(end),'completed',abs(out.state_log.Time(end)-P.duration)<1e-8);
        if isempty(results), results=item; else, results(end+1)=item; end %#ok<AGROW>
        fprintf('SLX_CANDIDATE %d %s rmse=%.6f max=%.6f final=%.6f tilt=%.2f command=%.1f completed=%d\n',...
            candidate,item.name,item.rmse_m,item.maximum_error_m,item.final_error_m,item.maximum_tilt_deg,...
            item.maximum_command_rad_s,item.completed);
        fid=fopen(fullfile(folder,'candidate_results.json'),'w'); fprintf(fid,'%s',jsonencode(results,PrettyPrint=true)); fclose(fid);
    end
end
save(fullfile(folder,'candidate_results.mat'),'results');
    function apply(P)
        ws=get_param(mdl,'ModelWorkspace'); ws.assignin('P',P);
        groups=hex_block_parameters(P); fields=fieldnames(groups);
        for j=1:numel(fields), ws.assignin(['P_' fields{j}],groups.(fields{j})); end
    end
end
function data=values(ts)
if ts.IsTimeFirst, data=reshape(ts.Data,numel(ts.Time),[]); else, data=reshape(ts.Data,[],numel(ts.Time))'; end
end
function close_model(mdl)
if bdIsLoaded(mdl), set_param(mdl,'FastRestart','off'); close_system(mdl,0); end
end
