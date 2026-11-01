function report=validate_hex_components()
%VALIDATE_HEX_COMPONENTS Independent causal/latency observer checks.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'blocks'),fullfile(root,'functions'));
P=hex_parameters(); groups=hex_block_parameters(P);
for fields={fieldnames(groups.controller_step),fieldnames(groups.task_reference)}
    assert(~any(ismember(fields{1},{'fault_motor','fault_factor','fault_start','fault_end','effectiveness'})),...
        'A control block received ground-truth fault parameters.');
end
checks=struct([]);
for delay=[0 .01 .03]
    for eta=[.8 .2 0]
        clear controller_step
        Q=P; Q.observed_feedback=true; Q.feedback_delay=delay;
        Q=hex_block_parameters(Q); C=Q.controller_step;
        state=P.initial_state; count=301; dt=P.control_dt; d=round(delay/dt);
        rpm=repmat(state(14:19),1,count); commands=zeros(6,count);
        decay=exp(-dt./P.tau); reference=[state(1:3);zeros(3,1)];
        for k=1:count
            observation=state; observation(14:19)=rpm(:,max(1,k-d));
            [commands(:,k),audit]=controller_step(observation,reference,C);
            beta=ones(6,1); if (k-1)*dt>=1, beta(2)=sqrt(eta); end
            if k<count, rpm(:,k+1)=decay.*rpm(:,k)+(1-decay).*beta.*commands(:,k); end
        end
        estimate=audit(44:49); expected=ones(6,1); expected(2)=eta;
        error=max(abs(estimate-expected)); assert(error<1e-5,'RPM observer or command-delay alignment failed.');
        item=struct('delay_s',delay,'factor',eta,'maximum_eta_error',error,'passed',true);
        if isempty(checks), checks=item; else, checks(end+1)=item; end %#ok<AGROW>
    end
end
clear reference_step
Q=groups.task_reference; observation=P.initial_state;
phase=zeros(201,1); speed=phase;
for k=1:201
    if k>=51, observation(7:10)=[sin(pi/9);0;0;cos(pi/9)]; end
    [~,audit]=reference_step((k-1)*P.control_dt,observation,Q);
    phase(k)=audit(1); speed(k)=audit(2);
end
assert(all(diff(phase)>=-1e-12) && all(speed>=0 & speed<=1),'Noncausal trajectory progress.');
assert(speed(end)<1e-6,'Governor did not pause for a large measured tilt.');
report=struct('observer_checks',checks,'truth_isolation_passed',true,...
    'governor_pause_passed',true,'passed',true);
fid=fopen(fullfile(root,'validation','SIMULATION_COMPONENT_RESULTS.json'),'w');
fprintf(fid,'%s',jsonencode(report,PrettyPrint=true)); fclose(fid);
fprintf('SIMULATION_COMPONENTS_PASSED observer_cases=%d\n',numel(checks));
end
