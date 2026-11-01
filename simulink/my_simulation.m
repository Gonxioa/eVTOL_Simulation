simulation_folder = fileparts(mfilename('fullpath'));
addpath(simulation_folder);

% 当前故障恢复模型：稳定姿态/高度、返回轨迹，再继续任务。
P = hex_parameters();

% 设置任务
P.task_family = 2;     % 0悬停、1直线、2绕圈、3升降、4正弦跟踪、5阶跃、6饱和测试
P.duration = 24;       % 仿真总时长：秒
P.motion_start = 1;    % 任务运动开始时间
P.motion_end = 7;      % 名义路径进度终点；故障恢复时允许减慢或暂停进度
P.radius = 0.3;        % 半径：米
P.turn = 2*pi;         % 弧度
% 设置故障
P.fault_motor = 1;             % 0=健康工况；1～6故障电机编号
P.fault_factor = 0.2;          % 剩余效能；0.8=Partial，0.2=Severe，0=Full，1=无损失
P.fault_start = 3;             % 故障开始的仿真时间
P.fault_end = Inf;             % 故障结束的仿真时间

% 设置环境和反馈
P.observed_feedback = true;    % 使用传感器观测反馈
P.noise_enabled = true;        % 启用测量噪声
P.wind_enabled = true;         % 启用风场
P.feedback_delay = 0.03;       % 反馈延迟：秒

% 生成本次风和测量误差，再同步模型参数
P = hex_prepare_inputs(P);
configure_hex_simulink(P);

%绘图
motor_to_plot = 1:6;

r = run_hex_simulink(Parameters=P, OutputName="my_circle_run", ...
    SaveResults=true);
output_folder = fullfile(simulation_folder, 'output', 'simulations', ...
    P.control_tuning_version, 'my_circle_run');
plot_hex_results(r, output_folder, motor_to_plot);

fprintf('六旋翼模型 hex_evtol，实际仿真时长 %.2f 秒。\n', r.time(end));
fprintf('结果已保存到：%s\n', output_folder);
