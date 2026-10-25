function style_hex_model(mdl)
%STYLE_HEX_MODEL Keep the top-level signal diagram readable at normal zoom.
blocks=find_system(mdl,'SearchDepth',1,'Type','Block');
old={'simulation_time','record_state','record_reference','record_motor_command','record_imu','record_observation','record_wind','record_effectiveness','record_control_audit','record_feedback'};
new={'t','log_state','log_ref','log_cmd','log_imu','log_obs','log_wind','log_eta','log_audit','log_feedback'};
for i=1:numel(blocks)
    type=get_param(blocks{i},'BlockType');
    if strcmp(type,'SubSystem'), set_param(blocks{i},'ContentPreviewEnabled','off'); end
    if strcmp(type,'Goto') || strcmp(type,'From')
        tag=get_param(blocks{i},'GotoTag'); index=find(strcmp(old,tag),1);
        if ~isempty(index), tag=new{index}; set_param(blocks{i},'GotoTag',tag); end
        set_param(blocks{i},'ShowName','off','IconDisplay','Tag','FontSize','10');
        if strcmp(type,'Goto'), set_param(blocks{i},'ShowName','on','Name',tag); end
    end
end
end
