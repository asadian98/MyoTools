function Z = ExtractML_Myo(ML_filename)

%% MonkeyLogic data

predur_min = 300;
predur_max = 500;
postdur = 500;

[A,B]=butter(3,0.0918); % Butterworth coefficients for G. 3rd order lowpass filter with fc = 0.0918 fs, with fs = 1000, fc = 45.9. Emulates the old Usui and Amidror 1982 values.

[ML_behavioural_data,ML_Config,ML_TrialRecord] = mlread(ML_filename);
st_ML = [];
for i = 1:size(ML_behavioural_data, 2)
    st_ML = [st_ML; ML_behavioural_data(i).BehavioralCodes.CodeNumbers];
end
start_tr_num_list = 1;
end_tr_num_list = size(ML_behavioural_data,2);

% Creating Z structure
parainfo.perfcodes{1,1} = 1;    parainfo.perfcodes{1,2} = 'Correct';
parainfo.perfcodes{2,1} = 0;    parainfo.perfcodes{2,2} = 'Incorrect';
parainfo.perfcodes{3,1} = -1;   parainfo.perfcodes{3,2} = 'Auto rejected due to RT outside min/max';
parainfo.perfcodes{4,1} = -2;   parainfo.perfcodes{4,2} = 'Subject broke fixation';
parainfo.perfcodes{5,1} = -99;  parainfo.perfcodes{5,2} = 'User rejected';

keys = [4];  % keys are monkeylogic codes
values = struct('condition', {1}, 'description', ...
    {'SOA'});

% Initialize the map
conditionCodeMap = containers.Map('KeyType', 'double', 'ValueType', 'any');

for i = 1:length(keys)
    conditionCodeMap(keys(i)) = values(i);  % Assign struct as value for each key
end

keys = 1:6;  % keys are monkeylogic codes
values = struct('code', {9, 18, 10, 30, 41, 42}, 'description', ...
    {'Trial Start', 'Trial End', 'T0_EYE_ON', 'T0_EYE_OFF', 'T1_EYE_ON', ...
    'T2_EYE_ON'});

% Initialize the map
strobecodesMap = containers.Map('KeyType', 'double', 'ValueType', 'any');

for i = 1:length(keys)
    strobecodesMap(keys(i)) = values(i);  % Assign struct as value for each key
end

keys = strobecodesMap.keys;
for keys_idx = 1:length(keys)
    key = keys{keys_idx};
    value = strobecodesMap(key);  % Retrieve the struct
    parainfo.strobecodes{key,1} = value.code;  parainfo.strobecodes{key,2} = value.description;
end

ML_analoginfo.Eye{1,1} = 'Horizontal Eye'; ML_analoginfo.eye{1,2} = 'Vertical Eye';
ML_analoginfo.Eye2{1,1} = 'EMPTY'; ML_analoginfo.eye2{1,2} = 'EMPTY';
ML_analoginfo.EyeExtra{1,1} = 'EMPTY';  ML_analoginfo.EyeExtra{1,2} = 'EMPTY';
ML_analoginfo.Joystick{1,1} = 'EMPTY'; ML_analoginfo.Joystick{1,2} = 'EMPTY';
ML_analoginfo.Joystick2{1,1} = 'EMPTY'; ML_analoginfo.Joystick2{1,2} = 'EMPTY';
ML_analoginfo.Touch{1,1} = 'Horizontal Hand'; ML_analoginfo.Touch{1,2} = 'Vertical Hand';
ML_analoginfo.Mouse{1,1} = 'EMPTY'; ML_analoginfo.Mouse{1,2} = 'EMPTY';
ML_analoginfo.KeyInput{1,1} = 'EMPTY'; ML_analoginfo.KeyInput{1,2} = 'EMPTY';
ML_analoginfo.Photodiode{1,1} = 'EMPTY';
ML_analoginfo.Photodiode{1,1} = 'EMPTY';

ML_analoginfo.General.gen1{1,1}  = 'Photodiode associated with relevant target onset';
ML_analoginfo.General.gen2{1,1}  = 'EMPTY';
ML_analoginfo.General.gen3{1,1}  = 'EMPTY';
ML_analoginfo.General.gen4{1,1}  = 'EMPTY';
ML_analoginfo.General.gen5{1,1}  = 'EMPTY';
ML_analoginfo.General.gen6{1,1}  = 'EMPTY';
ML_analoginfo.General.gen7{1,1}  = 'EMPTY';
ML_analoginfo.General.gen8{1,1}  = 'EMPTY';
ML_analoginfo.General.gen9{1,1}  = 'EMPTY';
ML_analoginfo.General.gen10{1,1} = 'EMPTY';

ML_analoginfo.Button.btn1{1,1}  = 'EMPTY';
ML_analoginfo.Button.btn2{1,1}  = 'EMPTY';
ML_analoginfo.Button.btn3{1,1}  = 'EMPTY';
ML_analoginfo.Button.btn4{1,1}  = 'EMPTY';
ML_analoginfo.Button.btn5{1,1}  = 'EMPTY';
ML_analoginfo.Button.btn6{1,1}  = 'EMPTY';
ML_analoginfo.Button.btn7{1,1}  = 'EMPTY';
ML_analoginfo.Button.btn8{1,1}  = 'EMPTY';
ML_analoginfo.Button.btn9{1,1}  = 'EMPTY';
ML_analoginfo.Button.btn10{1,1} = 'EMPTY';

Z.parainfo = parainfo;
Z.ML_analoginfo = ML_analoginfo;
Z_tr_offset = 0;

%% Specify and Extract Variables from MonkeyLogic File

tr_num_list = start_tr_num_list:end_tr_num_list;
for tr_num = tr_num_list
    disp(['Processing MonkeyLogic Behavioural Data for trial number ',num2str(tr_num)])
    Z.TrialNumber(tr_num + Z_tr_offset) = ML_behavioural_data(tr_num).Trial;  % Actual Trial Number
    Z.Block(tr_num + Z_tr_offset) = ML_behavioural_data(tr_num).Block;        % Block Number
    Z.TrialWithinBlock(tr_num + Z_tr_offset) = ML_behavioural_data(tr_num).TrialWithinBlock; % Trial Within Block

    % Convert  TrialError in ML File to perfcodes defined above
    temp_TrialError = ML_behavioural_data(tr_num).TrialError;
    if temp_TrialError == 0 % Correct Trial
        Z.TrialError(tr_num + Z_tr_offset) = 1;
    elseif temp_TrialError == 1
        Z.TrialError(tr_num + Z_tr_offset) = 0;
    elseif temp_TrialError == 3
        Z.TrialError(tr_num + Z_tr_offset) = -2;
    elseif temp_TrialError == 4
        Z.TrialError(tr_num + Z_tr_offset) = -2;
    elseif temp_TrialError == 5
        Z.TrialError(tr_num + Z_tr_offset) = -2;
    elseif temp_TrialError == 9
        Z.TrialError(tr_num + Z_tr_offset) = -2;
    elseif temp_TrialError == 6
        Z.TrialError(tr_num + Z_tr_offset) = -2;
    elseif temp_TrialError == 2
        Z.TrialError(tr_num + Z_tr_offset) = -2;
    else
        disp(['Trial error not classified for trial number: ',num2str(tr_num)])
    end
    Z.StrobeCodeNumbers{tr_num + Z_tr_offset} = ML_behavioural_data(tr_num).BehavioralCodes.CodeNumbers; % Strobe Code Times
    Z.StrobeCodeTimes{tr_num + Z_tr_offset} = ML_behavioural_data(tr_num).BehavioralCodes.CodeTimes; % Strobe Code Numbers

    % Get all Trial Variables intantiated within a given trial
    fields_UserVars = fieldnames(ML_behavioural_data(tr_num).UserVars);
    for fn_UV = 1:length(fields_UserVars)
        TrialVars{fn_UV,1} = fields_UserVars{fn_UV};
        TrialVars{fn_UV,2} = getfield(ML_behavioural_data(tr_num).UserVars,fields_UserVars{fn_UV});
    end
    Z.TrialVars{tr_num + Z_tr_offset} = TrialVars;
    % Get all object parameters instantiated in a give trial
    fields_SceneParameters = fieldnames(ML_behavioural_data(tr_num).ObjectStatusRecord.SceneParam);
    for fn_SP = 1:length(fields_SceneParameters)
        SceneParameters{fn_SP,1} = fields_SceneParameters{fn_SP};
        SceneParameters{fn_SP,2} = getfield(ML_behavioural_data(tr_num).ObjectStatusRecord.SceneParam,fields_SceneParameters{fn_SP});
    end
    Z.SceneParameters{tr_num + Z_tr_offset} = SceneParameters;
    % Catalog original ML Trial Type and convert ML Trial Type to
    % correct "parainfo.condition"
    % NOTE: This list of trial numbers may change with each
    % paradigm/experiment!!!
%     if isfield(ML_behavioural_data(tr_num).UserVars,'Trial_Type')
%         ML_TrialTypeNumber = getfield(ML_behavioural_data(tr_num).UserVars,'Trial_Type');
%         Z.ML_TrialTypeNumber(tr_num + Z_tr_offset) = ML_TrialTypeNumber;
% 
%         % __________________________ Set Condition number -------------
%         % You are giving the function a container map which includes
%         % task code (the one you defined in ML), condition number (the
%         % one you want to use in your analysis) and description
%         key = Z.ML_TrialTypeNumber(tr_num + Z_tr_offset);
%         if isKey(conditionCodeMap, key)
%             value = conditionCodeMap(key);  % Retrieve the struct
%             Z.condition(tr_num + Z_tr_offset) = value.condition;
%         else
%             disp('You have an undefined condition in your Z structure')
%         end
% 
%         parainfo.paradigm = 'Ernie SOA';
%         parainfo.conditions{value.condition,1} = value.condition; parainfo.conditions{value.condition,2} =  value.description; % Saccade only task (trial type number: 502)
% 
%     end
    Z.condition = [ML_behavioural_data.Condition];

    % Get all ML AnalogInputs within a given trial
    fields_AnalogData = fieldnames(ML_behavioural_data(tr_num).AnalogData);
    for fn_AD = 1:length(fields_AnalogData)
        AnalogData{fn_AD,1} = fields_AnalogData{fn_AD};
        AnalogData{fn_AD,2} = getfield(ML_behavioural_data(tr_num).AnalogData,fields_AnalogData{fn_AD});
    end
    Z.ML_AnalogData{tr_num + Z_tr_offset} = AnalogData;

    % Splice out the data you want
    % Timeframe to splice data
    Z.time_re_target1 = -predur_min:postdur;
    Z.time_re_target2 = -predur_max:postdur;

    % find first time that photodiode went high based on
    % thresholded value
    photothreshold = 0.5; %mV
    photodiode_data = ML_behavioural_data(tr_num + Z_tr_offset).AnalogData.General.Gen1;
    photodiode_data_greaterthanthreshold = find(photodiode_data(71:end) > photothreshold,1) + 70;
    if ~isempty(photodiode_data_greaterthanthreshold)
        Z.target_on_diode1(tr_num + Z_tr_offset) = photodiode_data_greaterthanthreshold;
    else
        Z.target_on_diode1(tr_num + Z_tr_offset) = NaN;
    end

    photothreshold = 0.5; %mV
    photodiode_data = ML_behavioural_data(tr_num + Z_tr_offset).AnalogData.General.Gen1;
    photodiode_data_greaterthanthreshold = find(photodiode_data(71:end) > photothreshold,1) + 70;
    if ~isempty(photodiode_data_greaterthanthreshold)
        Z.target_on_diode2(tr_num + Z_tr_offset) = photodiode_data_greaterthanthreshold;
    else
        Z.target_on_diode2(tr_num + Z_tr_offset) = NaN;
    end

    Z.target_on_diode(tr_num + Z_tr_offset) = min(Z.target_on_diode2(tr_num + Z_tr_offset), Z.target_on_diode1(tr_num + Z_tr_offset));

    Z.target_on_diode(tr_num + Z_tr_offset) = ML_behavioural_data(tr_num + Z_tr_offset).BehavioralCodes.CodeTimes(find(ML_behavioural_data(3).BehavioralCodes.CodeNumbers == 40)) + 28;
    Z.target_on_diode1(tr_num + Z_tr_offset) = Z.target_on_diode(tr_num + Z_tr_offset);

    if(Z.condition(tr_num + Z_tr_offset) == 5 || Z.condition(tr_num + Z_tr_offset) == 6)
        predata_duration = predur_max;
    else
        predata_duration = predur_min;
    end

    % Get spliced Eye, Touch, and Photodiode

    if Z.TrialError(tr_num + Z_tr_offset) == 1

        Z.EyeX_Raw_Diode{tr_num + Z_tr_offset} = Z.ML_AnalogData{1,tr_num + Z_tr_offset}{2,2}(round(Z.target_on_diode(tr_num + Z_tr_offset))-predata_duration:end,1);
        Z.EyeY_Raw_Diode{tr_num + Z_tr_offset} = Z.ML_AnalogData{1,tr_num + Z_tr_offset}{2,2}(round(Z.target_on_diode(tr_num + Z_tr_offset))-predata_duration:end,2); % round(Z.target_on_diode(tr_num + Z_tr_offset))+postdata_duration

        % Filtered Eye
        Z.EyeX_Filt_Diode{tr_num + Z_tr_offset} = filtfilt(A,B,Z.EyeX_Raw_Diode{tr_num + Z_tr_offset});
        Z.EyeY_Filt_Diode{tr_num + Z_tr_offset} = filtfilt(A,B,Z.EyeY_Raw_Diode{tr_num + Z_tr_offset});

        % Photodiode
        Z.Photodiode_Diode{tr_num + Z_tr_offset} = photodiode_data(round(Z.target_on_diode(tr_num + Z_tr_offset))-predata_duration:end);

        % Touchscreen
        if (~isempty(Z.ML_AnalogData{1, 1}{7, 2}) && isempty(Z.ML_AnalogData{1, 1}{8, 2})) % touch data, no mouse data
            Z.TouchX_Diode{tr_num + Z_tr_offset} = Z.ML_AnalogData{1, tr_num + Z_tr_offset}{7, 2}(round(Z.target_on_diode(tr_num + Z_tr_offset))-predata_duration:end,1); % round(Z.target_on_diode(tr_num + Z_tr_offset))+postdata_duration
            Z.TouchY_Diode{tr_num + Z_tr_offset} = Z.ML_AnalogData{1, tr_num + Z_tr_offset}{7, 2}(round(Z.target_on_diode(tr_num + Z_tr_offset))-predata_duration:end,2); % round(Z.target_on_diode(tr_num + Z_tr_offset))+postdata_duration
        elseif (isempty(Z.ML_AnalogData{1, 1}{7, 2}) && ~isempty(Z.ML_AnalogData{1, 1}{8, 2})) % no touch data, mouse data
            Z.TouchX_Diode{tr_num + Z_tr_offset} = Z.ML_AnalogData{1, tr_num + Z_tr_offset}{8, 2}(round(Z.target_on_diode(tr_num + Z_tr_offset))-predata_duration:end,1); % round(Z.target_on_diode(tr_num + Z_tr_offset))+postdata_duration
            Z.TouchY_Diode{tr_num + Z_tr_offset} = Z.ML_AnalogData{1, tr_num + Z_tr_offset}{8, 2}(round(Z.target_on_diode(tr_num + Z_tr_offset))-predata_duration:end,2); % round(Z.target_on_diode(tr_num + Z_tr_offset))+postdata_duration
        end

        % Set GAP - TARGET LOCATION - TARGET_COLOUR
        Z = setTaskconstraint(Z, tr_num, Z_tr_offset);

    else
        % Raw Eye
        Z.EyeX_Raw_Diode{tr_num + Z_tr_offset} = NaN(1,length(Z.time_re_target1))';
        Z.EyeY_Raw_Diode{tr_num + Z_tr_offset} = NaN(1,length(Z.time_re_target1))';

        % Filtered Eye
        Z.EyeX_Filt_Diode{tr_num + Z_tr_offset} = NaN(1,length(Z.time_re_target1))';
        Z.EyeY_Filt_Diode{tr_num + Z_tr_offset} = NaN(1,length(Z.time_re_target1))';

        % Photodiode
        Z.Photodiode_Diode{tr_num + Z_tr_offset} = NaN(1,length(Z.time_re_target1))';

        % Touchscreen
        Z.TouchX_Diode{tr_num + Z_tr_offset} = NaN(1,length(Z.time_re_target1))';
        Z.TouchY_Diode{tr_num + Z_tr_offset} = NaN(1,length(Z.time_re_target1))';

        % Reaction Time
        Z.Reach_RT_Diode(tr_num + Z_tr_offset) = NaN;

        % Target Characteristics
        Z.Target_Colour{tr_num + Z_tr_offset} = 'NONE';
        Z.Target_Location{1,tr_num + Z_tr_offset} = NaN;
        Z.Target_Location{2,tr_num + Z_tr_offset} = 'NONE';

        Z.GapDur(tr_num + Z_tr_offset) = nan;
    end

end

Z = detect_eye_arm_Ripple(Z, predur_min, predur_max);

end 

function Z = setTaskconstraint(Z, tr_num, Z_tr_offset)

% Get Target Location/Colour Information

T0_X = cell2mat(Z.TrialVars{1, tr_num + Z_tr_offset}(find(strcmp(Z.TrialVars{1, tr_num + Z_tr_offset},'T0_X')),2));
T0_Y = cell2mat(Z.TrialVars{1, tr_num + Z_tr_offset}(find(strcmp(Z.TrialVars{1, tr_num + Z_tr_offset},'T0_Y')),2));
T1_X = cell2mat(Z.TrialVars{1, tr_num + Z_tr_offset}(find(strcmp(Z.TrialVars{1, tr_num + Z_tr_offset},'T1_X')),2));
T1_Y = cell2mat(Z.TrialVars{1, tr_num + Z_tr_offset}(find(strcmp(Z.TrialVars{1, tr_num + Z_tr_offset},'T1_Y')),2));

if T1_X > T0_X
    Z.Target_Location{1,tr_num + Z_tr_offset} = 1;
    Z.Target_Location{2,tr_num + Z_tr_offset} = 'Right';
elseif T1_X < T0_X
    Z.Target_Location{1,tr_num + Z_tr_offset} = -1;
    Z.Target_Location{2,tr_num + Z_tr_offset} = 'Left';
else
    Z.Target_Location{1,tr_num + Z_tr_offset} = 0;
    Z.Target_Location{2,tr_num + Z_tr_offset} = 'VertOnly';
end

if T1_Y > T0_Y
    Z.Target_Location{3,tr_num + Z_tr_offset} = 1;
    Z.Target_Location{4,tr_num + Z_tr_offset} = 'Up';
elseif T1_Y < T0_Y
    Z.Target_Location{3,tr_num + Z_tr_offset} = -1;
    Z.Target_Location{4,tr_num + Z_tr_offset} = 'Down';
else
    Z.Target_Location{3,tr_num + Z_tr_offset} = 0;
    Z.Target_Location{4,tr_num + Z_tr_offset} = 'HoriOnly';
end

Z.Target_Colour{tr_num + Z_tr_offset} = 'GREEN';


end