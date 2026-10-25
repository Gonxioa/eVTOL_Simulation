function results=tune_hex_pv_baseline()
%TUNE_HEX_PV_BASELINE Screen gains against unchanged position/velocity tasks.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'blocks'),fullfile(root,'functions'));
folder=fullfile(root,'output','baseline_tuning');
if ~isfolder(folder), mkdir(folder); end
base=hex_parameters();
% [position P, velocity P, attitude P, rate P, rate D]. Keep yaw unchanged.
gains=[1.2 2.5 4 8 .1; ...
       2 5 8 16 .15; ...
       3 6 10 20 .2; ...
       3 8 12 24 .2; ...
       4 10 14 28 .25; ...
       4 8 14 30 .3; ...
       3 8 16 40 .4; ...
       4 10 16 40 .5; ...
       5 12 18 45 .6];
results=struct([]);
for i=1:size(gains,1)
    P=base; g=gains(i,:);
    P.control.position_p(1:2)=g(1);
    P.control.velocity_p(1:2)=g(2);
    P.control.attitude_p(1:2)=g(3);
    P.control.rate_p(1:2)=g(4);
    P.control.rate_d(1:2)=g(5);
    P.task_family=2;
    item=flight(P);
    item.index=i; item.gains=g; item.control=P.control;
    if isempty(results), results=item; else, results(end+1)=item; end %#ok<AGROW>
    fprintf('SCREEN %d rmse=%.6f max=%.6f final=%.6f tilt=%.2f command=[%.1f %.1f] completed=%d\n',...
        i,item.rmse_m,item.maximum_error_m,item.final_error_m,item.maximum_tilt_deg,...
        item.minimum_command_rad_s,item.maximum_command_rad_s,item.completed);
    fid=fopen(fullfile(folder,'screen.json'),'w');
    fprintf(fid,'%s',jsonencode(results,PrettyPrint=true)); fclose(fid);
end
save(fullfile(folder,'screen.mat'),'results','base');
end

function item=flight(P)
clear controller_step
state=P.initial_state; times=(0:round(P.duration/P.base_dt))'*P.base_dt;
N=numel(times); errors=zeros(N,1); tilt=zeros(N,1); commands=zeros(N,6);
completed=true; last=N;
for k=1:N
    t=times(k); ref=task_reference(t,P);
    [command,~]=controller_step(state,ref,P);
    errors(k)=norm(state(1:3)-ref(1:3)); commands(k,:)=command';
    R=hex_rotation(state(7:10)); tilt(k)=acosd(min(max(R(3,3),-1),1));
    if envelope_step(state,P), completed=false; last=k; break; end
    if k<N
        [segments,~]=fault_schedule(t,P);
        state=plant_step(state,command,segments,zeros(3,1),zeros(6,1),P);
    end
end
item=struct('rmse_m',sqrt(mean(errors(1:last).^2)),...
    'maximum_error_m',max(errors(1:last)), 'final_error_m',errors(last),...
    'maximum_tilt_deg',max(tilt(1:last)),...
    'minimum_command_rad_s',min(commands(1:last,:),[],'all'),...
    'maximum_command_rad_s',max(commands(1:last,:),[],'all'),...
    'completed',completed,'actual_duration_s',times(last));
end
