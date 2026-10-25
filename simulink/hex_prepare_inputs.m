function P=hex_prepare_inputs(P)
%HEX_PREPARE_INPUTS Separate reproducible streams; no noise inside RK stages.
N=ceil(P.duration/P.sensor_dt)+2;
streams=cell(1,8);
for j=1:8
    streams{j}=RandStream('mt19937ar','Seed',mod(P.seed+104729*j,2^32));
end
P.exogenous=zeros(N,24); ba=zeros(3,1); bg=zeros(3,1);
z=P.wind.ou_stationary_std_m_s*randn(streams{8},3,1);
dt=P.sensor_dt; c=P.sensor;
decay=exp(-dt/P.wind.ou_correlation_time_s);
for k=1:N
    if P.wind_enabled
        if k>1
            z=decay*z+P.wind.ou_stationary_std_m_s*sqrt(1-decay^2)*randn(streams{8},3,1);
        end
        P.exogenous(k,1:3)=(P.wind.mean_m_s(:)+z)';
    end
    if P.noise_enabled
        ba=ba+c.accel_bias_diffusion*sqrt(dt)*randn(streams{1},3,1);
        bg=bg+c.gyro_bias_diffusion*sqrt(dt)*randn(streams{2},3,1);
        P.exogenous(k,4:6)=(ba+c.accel_noise_density/sqrt(dt)*randn(streams{3},3,1)+c.additional_accel_sample_std*randn(streams{7},3,1))';
        P.exogenous(k,7:9)=(bg+c.gyro_noise_density/sqrt(dt)*randn(streams{4},3,1))';
        P.exogenous(k,10:12)=c.position_sample_std_m*randn(streams{5},1,3);
        P.exogenous(k,13:15)=c.velocity_sample_std_m_s*randn(streams{5},1,3);
        P.exogenous(k,16:18)=c.attitude_sample_std_rad*randn(streams{5},1,3);
        P.exogenous(k,19:24)=c.rpm_sample_std_rad_s*randn(streams{6},1,6);
    end
end
end
