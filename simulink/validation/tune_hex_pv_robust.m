function results=tune_hex_pv_robust(sweep)
%TUNE_HEX_PV_ROBUST Reject gains which only work with ideal state feedback.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'blocks'),fullfile(root,'functions'));
folder=fullfile(root,'output','baseline_tuning'); base=hex_parameters();
if nargin<1, sweep=1; end
gains=[3 6 10 20 .2;3 8 12 24 .2;2.5 6 10 20 .1;3 7 10 20 .1;...
       3 8 10 20 .1;4 6 10 20 .1;4 8 10 20 .1;3 8 12 20 .1;...
       3 8 10 16 .1;3 7 8 16 .1;3 6 8 16 .1;4 8 8 16 .1;...
       2.5 6 8 16 .15;3 10 10 20 .1;4 10 12 24 .1;3 6 12 24 .1];
if sweep==2
    gains=[1.2 2.5 4 8 .1;3 6 8 12 .3;3 6 8 12 .5;3 6 8 16 .3;...
        3 6 8 16 .5;3 6 10 16 .3;3 6 10 16 .5;3 8 10 16 .3;...
        3 8 10 16 .5;4 8 10 16 .5;4 8 10 20 .5;3 8 10 20 .3;...
        3 8 10 20 .5;3 8 12 24 .5;4 8 12 24 .5;3 6 8 12 .8;...
        4 8 12 16 .5;3 8 8 12 .5];
end
if sweep==3
    base.duration=20;
    gains=[4 6 8 16 .5;3.5 6 8 16 .5;3 6.5 8 18 .4;...
        3.5 7 8 20 .5;3 7 8 20 .5;4 7 8 20 .5;...
        3 6 7 16 .4;3 6 7 16 .5;3 6 8 18 .6;3.5 6.5 8 20 .5];
end
tag=sprintf('robust_screen_%d',sweep);
names={'ideal','observed_10ms','noise_wind_10ms','noise_wind_30ms'};
results=struct([]);
for i=1:size(gains,1)
    P=base; g=gains(i,:); P.task_family=2;
    P.control.position_p(1:2)=g(1); P.control.velocity_p(1:2)=g(2);
    P.control.attitude_p(1:2)=g(3); P.control.rate_p(1:2)=g(4); P.control.rate_d(1:2)=g(5);
    for condition=1:4
        Q=P; Q.observed_feedback=condition>=2;
        Q.noise_enabled=condition>=3; Q.wind_enabled=condition>=3;
        if condition==4, Q.feedback_delay=.03; end
        Q=hex_prepare_inputs(Q);
        item=flight(Q); item.index=i; item.gains=g; item.condition=names{condition}; item.control=P.control;
        if isempty(results), results=item; else, results(end+1)=item; end %#ok<AGROW>
        fprintf('ROBUST %d %s rmse=%.5f max=%.5f final=%.5f tilt=%.2f commands=[%.1f %.1f] completed=%d\n',...
            i,item.condition,item.rmse_m,item.maximum_error_m,item.final_error_m,item.maximum_tilt_deg,...
            item.minimum_command_rad_s,item.maximum_command_rad_s,item.completed);
    end
    fid=fopen(fullfile(folder,[tag '.json']),'w'); fprintf(fid,'%s',jsonencode(results,PrettyPrint=true)); fclose(fid);
end
save(fullfile(folder,[tag '.mat']),'results','base');
end
function item=flight(P)
clear controller_step sensor_step
state=P.initial_state; times=(0:round(P.duration/P.base_dt))'*P.base_dt;
N=numel(times); errors=zeros(N,1); tilt=zeros(N,1); commands=zeros(N,6);
completed=true; last=N;
for k=1:N
    t=times(k); ref=task_reference(t,P); [wind,loads]=wind_step(t,P);
    [feedback,~,~]=sensor_step(t,state,wind,loads,P);
    [command,~]=controller_step(feedback,ref,P);
    errors(k)=norm(state(1:3)-ref(1:3)); commands(k,:)=command';
    R=hex_rotation(state(7:10)); tilt(k)=acosd(min(max(R(3,3),-1),1));
    if envelope_step(state,P), completed=false; last=k; break; end
    if k<N
        [segments,~]=fault_schedule(t,P);
        state=plant_step(state,command,segments,wind,loads,P);
    end
end
item=struct('rmse_m',sqrt(mean(errors(1:last).^2)),...
    'rmse_first8_m',sqrt(mean(errors(1:min(last,801)).^2)),...
    'late_rmse_m',sqrt(mean(errors(max(1,last-199):last).^2)),...
    'maximum_error_m',max(errors(1:last)), 'final_error_m',errors(last),...
    'maximum_tilt_deg',max(tilt(1:last)),...
    'minimum_command_rad_s',min(commands(1:last,:),[],'all'),...
    'maximum_command_rad_s',max(commands(1:last,:),[],'all'),...
    'completed',completed,'actual_duration_s',times(last));
end
