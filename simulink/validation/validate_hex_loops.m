function report=validate_hex_loops()
%VALIDATE_HEX_LOOPS Isolated rotational plant, then attitude loop.
% This is a simulated loop bench: translation is constrained. For the rate
% bench only, a synthetic quaternion requests a known rate while keeping the
% production rate controller, estimator, allocator and motor dynamics intact.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'blocks'),fullfile(root,'functions'));
folder=fullfile(root,'output','loop_tests'); if ~isfolder(folder), mkdir(folder); end
P=hex_parameters(); P.observed_feedback=true; P.feedback_delay=.03;
P.noise_enabled=false; P.wind_enabled=false; P.duration=4;
results=struct([]);
for mode=1:2
    for axis=1:2
        candidates=[16 20 24]; if mode==2, candidates=[4 5 6 7 8]; end
        for gain=candidates
            clear controller_step
            Q=P;
            if mode==1, Q.control.rate_p(1:2)=gain;
            else, Q.control.attitude_p(1:2)=gain; end
            % Outer integral is disabled only on this constrained bench.
            Q.control.velocity_i=zeros(3,1);
            groups=hex_block_parameters(Q); C=groups.controller_step;
            state=Q.initial_state; N=401; states=repmat(state,1,N);
            rates=zeros(N,1); outputs=rates; desired=rates; commands=zeros(N,6);
            d=round(Q.feedback_delay/Q.control_dt);
            amplitude=.4; if mode==2, amplitude=5*pi/180; end
            for k=1:N
                t=(k-1)*Q.control_dt;
                setpoint=amplitude*double(t>=1 && t<2.5);
                obs=states(:,max(1,k-d)); obs(1:6)=[Q.task_origin;zeros(3,1)];
                reference=[Q.task_origin;zeros(3,1)];
                if mode==1
                    obs(7:10)=[0;0;0;1];
                    obs(6+axis)=-setpoint/(2*C.control.attitude_p(axis));
                    obs(10)=sqrt(1-sum(obs(7:9).^2));
                else
                    acceleration=zeros(3,1);
                    if axis==1, acceleration(2)=-Q.gravity*tan(setpoint);
                    else, acceleration(1)=Q.gravity*tan(setpoint); end
                    reference(1:3)=reference(1:3)+acceleration./(C.control.position_p.*C.control.velocity_p);
                end
                [command,audit]=controller_step(obs,reference,C);
                commands(k,:)=command'; desired(k)=setpoint;
                rates(k)=state(10+axis);
                R=hex_rotation(state(7:10));
                if axis==1, angle=atan2(R(3,2),R(3,3)); else, angle=asin(-R(3,1)); end
                outputs(k)=angle; if mode==1, outputs(k)=rates(k); end
                if k<N
                    segments=zeros(7,3); segments(:,1)=[Q.base_dt;ones(6,1)];
                    state=plant_step(state,command,segments,zeros(3,1),zeros(6,1),Q);
                    state(1:6)=[Q.task_origin;zeros(3,1)];
                    if mode==1, state(7:10)=[0;0;0;1]; end
                    states(:,k+1)=state;
                end
            end
            time=(0:N-1)'*Q.control_dt;
            positive=time>=1 & time<2.5; y=outputs(positive); tp=time(positive)-1;
            overshoot=max(0,(max(y)/amplitude-1)*100);
            outside=find(abs(y-amplitude)>.05*amplitude,1,'last');
            tenth=find(y>=.1*amplitude,1); ninetieth=find(y>=.9*amplitude,1);
            rise=Inf;
            if ~isempty(tenth) && ~isempty(ninetieth), rise=tp(ninetieth)-tp(tenth); end
            settling=0; if ~isempty(outside), settling=tp(outside); end
            if outside==numel(y), settling=Inf; end
            selected=(mode==1 && gain==20) || (mode==2 && gain==7);
            item=struct('mode',mode,'axis',axis,'rate_p',Q.control.rate_p(1),...
                'attitude_p',Q.control.attitude_p(1),...
                'rise_10_90_s',rise,...
                'overshoot_percent',overshoot,'settling_5percent_s',settling,...
                'rmse',sqrt(mean((y-amplitude).^2)),...
                'selected',selected,'passed',overshoot<=35 && settling<=1.2);
            if isempty(results), results=item; else, results(end+1)=item; end %#ok<AGROW>
            save(fullfile(folder,sprintf('mode%d_axis%d_gain%d.mat',mode,axis,gain)),...
                'time','outputs','desired','rates','commands','item');
            fprintf('SIMULATION_LOOP mode=%d axis=%d gain=%d overshoot=%.2f settling=%.2f passed=%d\n',...
                mode,axis,gain,overshoot,settling,item.passed);
        end
    end
end
report=struct('bench','translation-constrained rotational simulation, 30 ms measured-state delay',...
    'criteria',struct('maximum_overshoot_percent',35,'maximum_settling_5percent_s',1.2),...
    'cases',results,'passed',all([results([results.selected]).passed]));
fid=fopen(fullfile(root,'validation','SIMULATION_LOOP_RESULTS.json'),'w');
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
assert(report.passed,'Retained inner-loop gains failed the loop bench.');
end
