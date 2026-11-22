function identity=hex_run_identity(model,parameters)
%HEX_RUN_IDENTITY Snapshot the built model and repository revision per run.
root=fileparts(mfilename('fullpath'));
identity=hex_check_loaded_model(model);
identity.model_path=get_param(model,'FileName');
identity.model_sha256=hex_hash_file(identity.model_path);
[status,commit]=system(sprintf('git -C "%s" rev-parse HEAD',fileparts(root)));
assert(status==0,'hex:GitIdentity','Cannot determine the source commit.');
identity.git_commit=strtrim(commit);
[status,changes]=system(sprintf('git -C "%s" status --porcelain --untracked-files=no',fileparts(root)));
identity.tracked_worktree_dirty=status~=0 || ~isempty(strtrim(changes));
identity.controller_version=parameters.control_tuning_version;
identity.run_started_utc=char(datetime('now','TimeZone','UTC'));
identity.fast_restart=false;
end
