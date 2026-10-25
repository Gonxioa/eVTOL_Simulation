function model_path=build_hex_model()
%BUILD_HEX_MODEL Build a portable SLX with the numerical code embedded in it.
root=fileparts(mfilename('fullpath')); mdl='hex_evtol';
P=hex_prepare_inputs(hex_parameters());
if bdIsLoaded(mdl), close_system(mdl,0); end
new_system(mdl); load_system('simulink');
set_param(mdl,'SolverType','Fixed-step','Solver','FixedStepDiscrete',...
    'FixedStep','P.base_dt','StopTime','P.duration','ReturnWorkspaceOutputs','on',...
    'SignalLogging','off','SaveTime','on','SaveOutput','off',...
    'Description',['Six-rotor research simulation migrated from a6b9f1dd3d839416e469755490b8ff7733acd4ee. ',...
    'ENU world, FLU body, Hamilton xyzw, SI units. Embedded MATLAB Function code; no Python runtime. ',...
    '100 Hz controller/sensors, 500 Hz RK4 substeps; fault truth enters plant only. ',...
    'PV reference: position/velocity only; internal fixed heading; no acceleration or yaw-rate feedforward.']);
set_param(mdl,'Description',[get_param(mdl,'Description') newline newline ...
    fileread(fullfile(root,'THIRD_PARTY_NOTICES.md')) newline newline fileread(fullfile(root,'LICENSE.md'))]);
ws=get_param(mdl,'ModelWorkspace'); ws.assignin('P',P);
groups=hex_block_parameters(P); group_names=fieldnames(groups);
for g=1:numel(group_names), ws.assignin(['P_' group_names{g}],groups.(group_names{g})); end
add_block('simulink/Sources/Digital Clock',[mdl '/Simulation Clock'],'SampleTime','P.base_dt','Position',[20 70 65 100]);
add_block('simulink/Signal Routing/Goto',[mdl '/Clock Tag'],'GotoTag','simulation_time','Position',[85 72 145 98]);
add_line(mdl,'Simulation Clock/1','Clock Tag/1');
make_subsystem('Task Reference',{'time'},{'reference'},'task_reference',{},'P.control_dt',[100 160 275 225],[.78 .90 1]);
make_subsystem('Flight Controller',{'feedback','reference'},{'motor_command','control_audit'},'controller_step',...
    {'hex_rotation','hex_multiply','hex_desired_attitude','hex_allocate','hex_trf'},'P.control_dt',[390 150 590 250],[.78 1 .84]);
make_subsystem('Fault Injection',{'time'},{'event_segments','effectiveness'},'fault_schedule',{},'P.base_dt',[655 30 850 100],[1 .85 .72]);
make_subsystem('Six Rotor Plant',{'state','motor_command','event_segments','wind','body_loads'},{'next_state'},'plant_step',...
    {'hex_derivative','hex_rotation','hex_multiply'},'P.base_dt',[675 155 875 300],[.78 .90 1]);
make_subsystem('Sensors and Feedback',{'time','state','wind','body_loads'},{'feedback','observation','imu_sensor'},'sensor_step',...
    {'hex_derivative','hex_rotation','hex_multiply'},'P.sensor_dt',[370 405 590 520],[1 .96 .75]);
make_subsystem('Wind and Body Loads',{'time'},{'wind','body_loads'},'wind_step',{},'P.sensor_dt',[100 390 280 480],[.89 .84 1]);
make_subsystem('Envelope Monitor',{'state'},{'stop'},'envelope_step',{},'P.base_dt',[1000 35 1190 100],[1 .85 .72]);
add_block('simulink/Discrete/Unit Delay',[mdl '/State Memory'],...
    'InitialCondition','P.initial_state','SampleTime','P.base_dt','Position',[985 195 1050 245]);
add_block('simulink/Sinks/Stop Simulation',[mdl '/Physical Envelope Exit'],'Position',[1240 55 1280 85]);
clock_to('Task Reference',1,[20 180 75 200]);
clock_to('Fault Injection',1,[555 57 610 77]);
clock_to('Wind and Body Loads',1,[20 422 75 442]);
clock_to('Sensors and Feedback',1,[285 415 340 435]);
wire('Task Reference/1','Flight Controller/2','reference p v');
wire('Sensors and Feedback/1','Flight Controller/1','measured feedback');
wire('Flight Controller/1','Six Rotor Plant/2','command rad per s');
wire('Fault Injection/1','Six Rotor Plant/3','plant fault schedule');
wire('Wind and Body Loads/1','Six Rotor Plant/4','wind ENU');
wire('Wind and Body Loads/2','Six Rotor Plant/5','body force and torque');
wire('Wind and Body Loads/1','Sensors and Feedback/3','');
wire('Wind and Body Loads/2','Sensors and Feedback/4','');
wire('Six Rotor Plant/1','State Memory/1','RK4 state at next tick');
wire('State Memory/1','Six Rotor Plant/1','physical state 19');
wire('State Memory/1','Sensors and Feedback/2','');
wire('State Memory/1','Envelope Monitor/1','');
wire('Envelope Monitor/1','Physical Envelope Exit/1','exit envelope');
sources={'State Memory/1','Task Reference/1','Flight Controller/1','Sensors and Feedback/3',...
    'Sensors and Feedback/2','Wind and Body Loads/1','Fault Injection/2','Flight Controller/2','Sensors and Feedback/1'};
names={'state','reference','motor_command','imu','observation','wind','effectiveness','control_audit','feedback'};
tag_positions={[1095 270 1215 290],[285 260 365 280],[600 115 715 135],...
    [605 550 740 570],[605 520 740 540],[105 510 265 530],...
    [865 105 1000 125],[600 320 740 340],[275 325 390 345]};
add_block('built-in/Subsystem',[mdl '/Flight Recorder'],'Position',[1000 365 1200 575],'BackgroundColor','[0.87 0.94 0.94]');
rec=[mdl '/Flight Recorder'];
for i=1:numel(names)
    port=[rec '/' names{i}];
    add_block('simulink/Ports & Subsystems/In1',port,'Port',num2str(i),'Position',[25 35*i 55 35*i+14]);
    sink=[rec '/' names{i} '_log'];
    add_block('simulink/Sinks/To Workspace',sink,'VariableName',[names{i} '_log'],...
        'SaveFormat','Timeseries','MaxDataPoints','inf','SampleTime','P.export_dt','Position',[180 35*i-5 315 35*i+20]);
    add_line(rec,[names{i} '/1'],[names{i} '_log/1']);
    tag=['record_' names{i}];
    add_block('simulink/Signal Routing/Goto',[mdl '/Record ' names{i}],...
        'GotoTag',tag,'Position',tag_positions{i});
    add_line(mdl,sources{i},['Record ' names{i} '/1'],'autorouting','on');
    add_block('simulink/Signal Routing/From',[mdl '/Read ' names{i}],...
        'GotoTag',tag,'Position',[845 365+23*i 960 380+23*i]);
    add_line(mdl,['Read ' names{i} '/1'],['Flight Recorder/' num2str(i)],'autorouting','on');
end
% Scope views live next to the logged signals inside Flight Recorder.
add_block('simulink/Signal Routing/Demux',[rec '/State Channels'],'Outputs','[3 3 4 3 6]','Position',[370 35 375 180]);
add_block('simulink/Signal Routing/Demux',[rec '/Reference Position'],'Outputs','[3 3]','Position',[370 205 375 250]);
add_line(rec,'state/1','State Channels/1','autorouting','on');
add_line(rec,'reference/1','Reference Position/1','autorouting','on');
add_block('simulink/Sinks/Scope',[rec '/Position Tracking'],'NumInputPorts','2','Position',[480 45 550 95]);
add_block('simulink/Sinks/Scope',[rec '/Rotor Speed and Command'],'NumInputPorts','2','Position',[480 140 550 190]);
add_block('simulink/Sinks/Scope',[rec '/IMU Measurements'],'Position',[480 245 550 290]);
add_line(rec,'State Channels/1','Position Tracking/1','autorouting','on');
add_line(rec,'Reference Position/1','Position Tracking/2','autorouting','on');
add_line(rec,'State Channels/5','Rotor Speed and Command/1','autorouting','on');
add_line(rec,'motor_command/1','Rotor Speed and Command/2','autorouting','on');
add_line(rec,'imu/1','IMU Measurements/1','autorouting','on');
note=Simulink.Annotation(mdl,sprintf('HEXAROTOR RESEARCH MODEL | a6b9f1d\nENU / FLU | SI | quaternion [x y z w]\nDouble-click subsystems to inspect embedded MATLAB code.\nModel Workspace P contains parameters and noise inputs.'));
note.Position=[110 15]; note.FontSize=12;
set_param(mdl,'Location',[50 60 1490 850],'ZoomFactor','FitSystem');
style_hex_model(mdl);
model_path=fullfile(root,[mdl '.slx']); save_system(mdl,model_path);
fprintf('MODEL_SAVED %s\n',model_path);

    function make_subsystem(name,inputs,outputs,source,helpers,sample,position,color)
        parent=[mdl '/' name];
        add_block('built-in/Subsystem',parent,'Position',position,'BackgroundColor',mat2str(color));
        block=[parent '/Embedded MATLAB'];
        add_block('simulink/User-Defined Functions/MATLAB Function',block,'Position',[180 60 405 190]);
        rt=sfroot; chart=rt.find('-isa','Stateflow.EMChart','Path',block);
        script=fileread(fullfile(root,'blocks',[source '.m']));
        for h=1:numel(helpers)
            script=[script newline fileread(fullfile(root,'functions',[helpers{h} '.m']))]; %#ok<AGROW>
        end
        parameter_name=['P_' source];
        chart.Script=regexprep(script,'\<P\>',parameter_name);
        data=chart.find('-isa','Stateflow.Data','Name',parameter_name); data.Scope='Parameter'; data.Tunable=false;
        chart.ChartUpdate='DISCRETE'; chart.SampleTime=sample;
        for j=1:numel(inputs)
            add_block('simulink/Ports & Subsystems/In1',[parent '/' inputs{j}],...
                'Port',num2str(j),'Position',[35 45*j 65 45*j+14]);
            add_line(parent,[inputs{j} '/1'],['Embedded MATLAB/' num2str(j)],'autorouting','on');
        end
        for j=1:numel(outputs)
            add_block('simulink/Ports & Subsystems/Out1',[parent '/' outputs{j}],...
                'Port',num2str(j),'Position',[480 65*j 510 65*j+14]);
            add_line(parent,['Embedded MATLAB/' num2str(j)],[outputs{j} '/1'],'autorouting','on');
        end
    end
    function clock_to(name,port,position)
        key=['Clock for ' name];
        add_block('simulink/Signal Routing/From',[mdl '/' key],'GotoTag','simulation_time','Position',position);
        add_line(mdl,[key '/1'],[name '/' num2str(port)],'autorouting','on');
    end
    function wire(from,to,label)
        line=add_line(mdl,from,to,'autorouting','on');
        if ~isempty(label), set_param(line,'Name',label); end
    end
end
