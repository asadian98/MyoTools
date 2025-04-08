clc; clear;

folder_path = 'C:\Users\CorneilLab\Desktop\MayoTools';
addpath(genpath(folder_path));

folder_path = 'C:\Users\CorneilLab\Desktop\SortingTools';
addpath(genpath(folder_path));

folder_path = 'C:\Users\CorneilLab\Desktop\SharedUtils';
addpath(genpath(folder_path));

folder_path = 'C:\Users\CorneilLab\Documents\NIMH_MonkeyLogic_2.2';
addpath(genpath(folder_path));

sesName =                   '2025-03-26_14-30-06';
animalName =                'Trex';

Folders.rec_path = ['C:\Users\CorneilLab\Desktop\Myo_Data\Raw_data\', animalName,'\', sesName, '\'];
cd(Folders.rec_path)

load('rec_Analog_2.mat')
load('rec_EMG_1k_2.mat')
load("rec_events_2.mat")
load('rec_Z_2.mat')

sampling_rate = 30000;

%% Calibrate Eye

st9 = find(rec_events{1}(1, :) == 9);
st18 = find(rec_events{1}(1, :) == 18);

matches = regexp(sesName, '(\d{4})-(\d{2})-(\d{2})', 'tokens');
year_last2 = matches{1}{1}(3:4); % Extract '25' from '2025'
result = [year_last2 matches{1}{2} matches{1}{3}]; % Concatenates '25', '03', '06'

[ML_behavioural_data,ML_Config,ML_TrialRecord] = mlread([result, '_', animalName, '_Myo_2.bhv2']);

tr = 100;
endd = 500;
EyeMLx_tr = ML_behavioural_data(tr).AnalogData.Eye(1:endd, 1)';
EyeMLy_tr = ML_behavioural_data(tr).AnalogData.Eye(1:endd, 2)';

sts = rec_events{1, 1}(2, find(rec_events{1, 1}(1, :) == 9));
st = sts(tr);
ens = rec_events{1, 1}(2, find(rec_events{1, 1}(1, :) == 18));
en = ens(tr);

Analog_Data_mV = double(rec_Analog{1, 1}.data);

st_samp = dsearchn(rec_Analog{1, 1}.timestamps', st);
en_samp = dsearchn(rec_Analog{1, 1}.timestamps', en);

EyeRPx_tr = Analog_Data_mV(1, st_samp:st_samp+endd-1);
EyeRPy_tr = Analog_Data_mV(2, st_samp:st_samp+endd-1);

EyeRPx = double(Analog_Data_mV(1, :));
EyeRPy = double(Analog_Data_mV(2, :));

figure
subplot(211)
plot(EyeRPx_tr)
subplot(212)
plot(EyeMLx_tr)

[x_calibrated, y_calibrated] = calibrateEyeRipple(EyeRPx_tr, EyeRPy_tr, EyeMLx_tr, EyeMLy_tr, EyeRPx, EyeRPy);

figure
plot(x_calibrated(st_samp:st_samp+endd-1)); 
hold on; 
plot(EyeMLx_tr)

%% Plot RT histogram

close all
cd('figures')

Z.TrialError(find(Z.EyeRT(:, 1) < 50 | Z.EyeRT(:, 1) > 500)) = 3;
histogram(Z.EyeRT(find(Z.TrialError == 1), 1), 100)
saveas(gcf, ['EyeRT_histogram' '.png']);

cd('..')
close all

%% Plot filtered emg signal over channels

EMG = rec_EMG_1k{1, 1}.data;

figure

for ch_ind = 14
    hold on
    plot(EMG(ch_ind, :) - (ch_ind-1)*50000)

end

%% Plot EMG over channels

offset = 0; % Initial vertical offset
vertical_shift = -1.1; % Amount to shift each plot vertically
figure;

for kk = 1:size(EMG, 1)

    data = EMG(kk, :); % Use the first channel of data

    % Plot with vertical shift
    plot(normalize(data - mean(data),'range', [0, 1])+offset, 'LineWidth', 1.5);

    hold on
    % Update the offset for the next stream
    offset = offset + vertical_shift;
end
title('EMG')

%%
sac_en = 2000;

PD1_Onsets = find(diff(Analog_Data_mV(4, :)) > 2000);
PD1_Onsets = PD1_Onsets([true, diff(PD1_Onsets) > 20]);

% target_onset_time = rec_events{1, 1}(2, find(rec_events{1, 1}(1, :) == 40));
% target_onset_samp = dsearchn(rec_Analog{1, 1}.timestamps', target_onset_time')+28; %28 is an estimate of difference between ML code and photodiode onset
% PD1_Onsets = target_onset_samp';

pre = 1000;
post = 1000;

[A,B]=butter(3,0.0918);
EyeX = filtfilt(A,B, x_calibrated');
EyeY = filtfilt(A,B, y_calibrated');

st = 0;
en = 1000;
Eye_RT = zeros(1, length(PD1_Onsets));

sts = rec_events{1, 1}(2, find(rec_events{1, 1}(1, :) == 9));

PD_ru = [];
PD_lu = [];
PD_rd = [];
PD_ld = [];
PD_rh = [];
PD_lh = [];

for PDj = 1:length(PD1_Onsets)

    PD = PD1_Onsets(PDj);
    PD_time = rec_Analog{1, 1}.timestamps(PD);
    tmp = find(sts < PD_time);tr = tmp(end);

    Eh = EyeX(PD+st:PD+en); % Horizontal Eye Position
    Ev = EyeY(PD+st:PD+en); % Vertical Eye Position
    dEh = diff(Eh')*1000;
    dEv = diff(Ev')*1000;

    output = sacdetector(dEh,dEv,Eh',Ev',[100, 50, 50],0);

%     if(~isempty(output) && Z.TrialError(tr) == 1  && Z.TrialVars{1, tr}{17, 2} == 0)
%         PD_rh = [PD_rh, PDj];
%     elseif(~isempty(output) && Z.TrialError(tr) == 1  && Z.TrialVars{1, tr}{17, 2} == 180)
%         PD_lh = [PD_lh, PDj];
%     end

    if(~isempty(output) && Z.TrialError(tr) == 1  && abs(Z.TrialVars{1, tr}{13, 2}) >= 10 && strcmp(Z.Target_Location{2, tr}, 'Right') && strcmp(Z.Target_Location{4, tr}, 'HoriOnly'))
        PD_rh = [PD_rh, PDj];
    elseif(~isempty(output) && Z.TrialError(tr) == 1 && abs(Z.TrialVars{1, tr}{13, 2}) >= 10 && strcmp(Z.Target_Location{2, tr}, 'Right') && strcmp(Z.Target_Location{4, tr}, 'Up'))
        PD_ru = [PD_ru, PDj];
    elseif(~isempty(output) && Z.TrialError(tr) == 1 && abs(Z.TrialVars{1, tr}{13, 2}) >= 10 && strcmp(Z.Target_Location{2, tr}, 'Right') && strcmp(Z.Target_Location{4, tr}, 'Down'))
        PD_rd = [PD_rd, PDj];
    elseif(~isempty(output) && Z.TrialError(tr) == 1 && abs(Z.TrialVars{1, tr}{13, 2}) >= 10 && strcmp(Z.Target_Location{2, tr}, 'Left') && strcmp(Z.Target_Location{4, tr}, 'HoriOnly'))
        PD_lh = [PD_lh, PDj];
    elseif(~isempty(output) && Z.TrialError(tr) == 1 && abs(Z.TrialVars{1, tr}{13, 2}) >= 10 && strcmp(Z.Target_Location{2, tr}, 'Left') && strcmp(Z.Target_Location{4, tr}, 'Up'))
        PD_lu = [PD_lu, PDj];
    elseif(~isempty(output) && Z.TrialError(tr) == 1 && abs(Z.TrialVars{1, tr}{13, 2}) >= 10 && strcmp(Z.Target_Location{2, tr}, 'Left') && strcmp(Z.Target_Location{4, tr}, 'Down'))
        PD_ld = [PD_ld, PDj];
    end

    if(~isempty(output) && Z.TrialError(tr) == 1)
        Eye_RT(PDj) = output(1, 3);
    end
end

%% data aligned to saccade for left and right saccades

time_pre_eye = 200;
time_post_eye = 200;

time_pre_tar = 200;
time_post_tar = 400;

EMG = rec_EMG_1k{1, 1}.data(1:16, :);

% EMG_aligned_r is index*ch*time
EMG_aligned_eye_ru = zeros(length(PD_ru), size(EMG, 1), time_pre_eye+time_post_eye+1);
EMG_aligned_eye_rd = zeros(length(PD_rd), size(EMG, 1), time_pre_eye+time_post_eye+1);
EMG_aligned_eye_rh = zeros(length(PD_rh), size(EMG, 1), time_pre_eye+time_post_eye+1);

Eye_aligned_eye_ru = zeros(length(PD_ru), 2, time_pre_eye+time_post_eye+1);
Eye_aligned_eye_rd = zeros(length(PD_rd), 2, time_pre_eye+time_post_eye+1);
Eye_aligned_eye_rh = zeros(length(PD_rh), 2, time_pre_eye+time_post_eye+1);

EMG_aligned_tar_ru = zeros(length(PD_ru), size(EMG, 1), time_pre_tar+time_post_tar+1);
EMG_aligned_tar_rd = zeros(length(PD_rd), size(EMG, 1), time_pre_tar+time_post_tar+1);
EMG_aligned_tar_rh = zeros(length(PD_rh), size(EMG, 1), time_pre_tar+time_post_tar+1);

Eye_aligned_tar_ru = zeros(length(PD_ru), 2, time_pre_tar+time_post_tar+1);
Eye_aligned_tar_rd = zeros(length(PD_rd), 2, time_pre_tar+time_post_tar+1);
Eye_aligned_tar_rh = zeros(length(PD_rh), 2, time_pre_tar+time_post_tar+1);

Analog_Data_cal = [x_calibrated; y_calibrated];

for PDj = 1:length(PD_ru)
    EyeRT_samp = PD1_Onsets(PD_ru(PDj))+Eye_RT(PD_ru(PDj));
    EMG_aligned_eye_ru(PDj, :, :) = EMG(:, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    Eye_aligned_eye_ru(PDj, :, :) = Analog_Data_cal(1:2, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    EMG_aligned_tar_ru(PDj, :, :) = EMG(:, PD1_Onsets(PD_ru(PDj))-time_pre_tar:PD1_Onsets(PD_ru(PDj))+time_post_tar);
    Eye_aligned_tar_ru(PDj, :, :) = Analog_Data_cal(1:2, PD1_Onsets(PD_ru(PDj))-time_pre_tar:PD1_Onsets(PD_ru(PDj))+time_post_tar);
    
end
for PDj = 1:length(PD_rd)
    EyeRT_samp = PD1_Onsets(PD_rd(PDj))+Eye_RT(PD_rd(PDj));
    EMG_aligned_eye_rd(PDj, :, :) = EMG(:, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    Eye_aligned_eye_rd(PDj, :, :) = Analog_Data_cal(1:2, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    EMG_aligned_tar_rd(PDj, :, :) = EMG(:, PD1_Onsets(PD_rd(PDj))-time_pre_tar:PD1_Onsets(PD_rd(PDj))+time_post_tar);
    Eye_aligned_tar_rd(PDj, :, :) = Analog_Data_cal(1:2, PD1_Onsets(PD_rd(PDj))-time_pre_tar:PD1_Onsets(PD_rd(PDj))+time_post_tar);
end
for PDj = 1:length(PD_rh)
    EyeRT_samp = PD1_Onsets(PD_rh(PDj))+Eye_RT(PD_rh(PDj));
    EMG_aligned_eye_rh(PDj, :, :) = EMG(:, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    Eye_aligned_eye_rh(PDj, :, :) = Analog_Data_cal(1:2, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    EMG_aligned_tar_rh(PDj, :, :) = EMG(:, PD1_Onsets(PD_rh(PDj))-time_pre_tar:PD1_Onsets(PD_rh(PDj))+time_post_tar);
    Eye_aligned_tar_rh(PDj, :, :) = Analog_Data_cal(1:2, PD1_Onsets(PD_rh(PDj))-time_pre_tar:PD1_Onsets(PD_rh(PDj))+time_post_tar);
end

EMG_aligned_eye_lu = zeros(length(PD_lu), size(EMG, 1), time_pre_eye+time_post_eye+1);
EMG_aligned_eye_ld = zeros(length(PD_ld), size(EMG, 1), time_pre_eye+time_post_eye+1);
EMG_aligned_eye_lh = zeros(length(PD_lh), size(EMG, 1), time_pre_eye+time_post_eye+1);

Eye_aligned_eye_lu = zeros(length(PD_lu), 2, time_pre_eye+time_post_eye+1);
Eye_aligned_eye_ld = zeros(length(PD_ld), 2, time_pre_eye+time_post_eye+1);
Eye_aligned_eye_lh = zeros(length(PD_lh), 2, time_pre_eye+time_post_eye+1);

EMG_aligned_tar_lu = zeros(length(PD_lu), size(EMG, 1), time_pre_tar+time_post_tar+1);
EMG_aligned_tar_ld = zeros(length(PD_ld), size(EMG, 1), time_pre_tar+time_post_tar+1);
EMG_aligned_tar_lh = zeros(length(PD_lh), size(EMG, 1), time_pre_tar+time_post_tar+1);

Eye_aligned_tar_lu = zeros(length(PD_lu), 2, time_pre_tar+time_post_tar+1);
Eye_aligned_tar_ld = zeros(length(PD_ld), 2, time_pre_tar+time_post_tar+1);
Eye_aligned_tar_lh = zeros(length(PD_lh), 2, time_pre_tar+time_post_tar+1);

for PDj = 1:length(PD_lu)
    EyeRT_samp = PD1_Onsets(PD_lu(PDj))+Eye_RT(PD_lu(PDj));
    EMG_aligned_eye_lu(PDj, :, :) = EMG(:, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    Eye_aligned_eye_lu(PDj, :, :) = Analog_Data_cal(1:2, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    EMG_aligned_tar_lu(PDj, :, :) = EMG(:, PD1_Onsets(PD_lu(PDj))-time_pre_tar:PD1_Onsets(PD_lu(PDj))+time_post_tar);
    Eye_aligned_tar_lu(PDj, :, :) = Analog_Data_cal(1:2, PD1_Onsets(PD_lu(PDj))-time_pre_tar:PD1_Onsets(PD_lu(PDj))+time_post_tar);
end
for PDj = 1:length(PD_ld)
    EyeRT_samp = PD1_Onsets(PD_ld(PDj))+Eye_RT(PD_ld(PDj));
    EMG_aligned_eye_ld(PDj, :, :) = EMG(:, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    Eye_aligned_eye_ld(PDj, :, :) = Analog_Data_cal(1:2, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    EMG_aligned_tar_ld(PDj, :, :) = EMG(:, PD1_Onsets(PD_ld(PDj))-time_pre_tar:PD1_Onsets(PD_ld(PDj))+time_post_tar);
    Eye_aligned_tar_ld(PDj, :, :) = Analog_Data_cal(1:2, PD1_Onsets(PD_ld(PDj))-time_pre_tar:PD1_Onsets(PD_ld(PDj))+time_post_tar);
end
for PDj = 1:length(PD_lh)
    EyeRT_samp = PD1_Onsets(PD_lh(PDj))+Eye_RT(PD_lh(PDj));
    EMG_aligned_eye_lh(PDj, :, :) = EMG(:, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    Eye_aligned_eye_lh(PDj, :, :) = Analog_Data_cal(1:2, EyeRT_samp-time_pre_eye:EyeRT_samp+time_post_eye);
    EMG_aligned_tar_lh(PDj, :, :) = EMG(:, PD1_Onsets(PD_lh(PDj))-time_pre_tar:PD1_Onsets(PD_lh(PDj))+time_post_tar);
    Eye_aligned_tar_lh(PDj, :, :) = Analog_Data_cal(1:2, PD1_Onsets(PD_lh(PDj))-time_pre_tar:PD1_Onsets(PD_lh(PDj))+time_post_tar);
end

%% data aligned to target onset / single channel / trial EMG

% cd('figures')
k = 15;
for ch_idx = 14

    figure
    subplot(1, 2, 1)
    plot(-time_pre_tar:time_post_tar, abs(reshape(EMG_aligned_tar_lh(:, ch_idx, :), size(EMG_aligned_tar_lh, 1), size(EMG_aligned_tar_lh, 3))'), 'k')
    hold on
    plot(-time_pre_tar:time_post_tar, mean(abs(reshape(EMG_aligned_tar_lh(:, ch_idx, :), size(EMG_aligned_tar_lh, 1), size(EMG_aligned_tar_lh, 3))), 1), 'r')
    ylim([0, mean(mean(EMG_aligned_tar_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_tar_rh(:, ch_idx, :), 0, 3))])
    xlim([-time_pre_tar, time_post_tar])
    title('Left')
    
    subplot(1, 2, 2)
    plot(-time_pre_tar:time_post_tar, abs(reshape(EMG_aligned_tar_rh(:, ch_idx, :), size(EMG_aligned_tar_rh, 1), size(EMG_aligned_tar_rh, 3))'), 'k')
    hold on
    plot(-time_pre_tar:time_post_tar, mean(abs(reshape(EMG_aligned_tar_rh(:, ch_idx, :), size(EMG_aligned_tar_rh, 1), size(EMG_aligned_tar_rh, 3))), 1), 'r')
    ylim([0, mean(mean(EMG_aligned_tar_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_tar_rh(:, ch_idx, :), 0, 3))])
    xlim([-time_pre_tar, time_post_tar])
    title('Right')

    sgtitle(['ch ', num2str(ch_idx)])
    
%     saveas(gcf, ['Trial_EMG_plot_aligned_tar_ch' num2str(ch_idx) '.png']);
end

for ch_idx = 14

    figure
    subplot(1, 2, 1)
    plot(-time_pre_eye:time_post_eye, abs(reshape(EMG_aligned_eye_lu(:, ch_idx, :), size(EMG_aligned_eye_lu, 1), size(EMG_aligned_eye_lu, 3))'), 'k')
    hold on
    plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_lu(:, ch_idx, :), size(EMG_aligned_eye_lu, 1), size(EMG_aligned_eye_lu, 3))), 1), 'r')
    ylim([0, mean(mean(EMG_aligned_eye_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_eye_rh(:, ch_idx, :), 0, 3))])
    xlim([-time_pre_eye, time_post_eye])
    title('Left')
    
    subplot(1, 2, 2)
    plot(-time_pre_eye:time_post_eye, abs(reshape(EMG_aligned_eye_rh(:, ch_idx, :), size(EMG_aligned_eye_rh, 1), size(EMG_aligned_eye_rh, 3))'), 'k')
    hold on
    plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_rh(:, ch_idx, :), size(EMG_aligned_eye_rh, 1), size(EMG_aligned_eye_rh, 3))), 1), 'r')
    ylim([0, mean(mean(EMG_aligned_eye_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_eye_rh(:, ch_idx, :), 0, 3))])
    xlim([-time_pre_eye, time_post_eye])
    title('Right')

    sgtitle(['ch ', num2str(ch_idx)])
    
    saveas(gcf, ['Trial_EMG_plot_aligned_eye_ch' num2str(ch_idx) '.png']);
end

% cd('..')
% close all

%% Plot trials in colors over time

cd('figures')

k = 15;

for ch_idx = 1:16

    time = -time_pre_tar:time_post_tar;
    
    data = abs(reshape(EMG_aligned_tar_lh(:, ch_idx, :), size(EMG_aligned_tar_lh, 1), size(EMG_aligned_tar_lh, 3)));
    
    f = figure;
    f.Position(3:4) = [1000 500];

    subplot(1, 2, 1)
    imagesc(time, 1:size(data, 1), data, [0, mean(mean(EMG_aligned_tar_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_tar_rh(:, ch_idx, :), 0, 3))]); % Plot matrix as image
    colormap('jet'); % Use 'jet' or any preferred colormap
    colorbar; % Add color scale
    xlabel('Time');
    ylabel('Trials');
    title('Trial Magnitude Over Time aligned to target onset');
    
    data = reshape(EMG_aligned_tar_rh(:, ch_idx, :), size(EMG_aligned_tar_rh, 1), size(EMG_aligned_tar_rh, 3));
    subplot(1, 2, 2)
    imagesc(time, 1:size(data, 1), data, [0, mean(mean(EMG_aligned_tar_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_tar_rh(:, ch_idx, :), 0, 3))]); % Plot matrix as image
    colormap('jet'); % Use 'jet' or any preferred colormap
    colorbar; % Add color scale
    xlabel('Time');
    ylabel('Trials');
    title('Trial Magnitude Over Time aligned to target onset');
    
    sgtitle(['Ch ', num2str(ch_idx)])
    saveas(gcf, ['Trial_EMG_color_plot_aligned_tar_ch' num2str(ch_idx) '.png']);

end

for ch_idx = 1:16

    time = -time_pre_eye:time_post_eye;
    
    data = abs(reshape(EMG_aligned_eye_lh(:, ch_idx, :), size(EMG_aligned_eye_lh, 1), size(EMG_aligned_eye_lh, 3)));
    
    f = figure;
    f.Position(3:4) = [1000 500];

    subplot(1, 2, 1)
    imagesc(time, 1:size(data, 1), data, [0, mean(mean(EMG_aligned_eye_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_eye_rh(:, ch_idx, :), 0, 3))]); % Plot matrix as image
    colormap('jet'); % Use 'jet' or any preferred colormap
    colorbar; % Add color scale
    xlabel('Time');
    ylabel('Trials');
    title('Trial Magnitude Over Time aligned to saccade onset');
    
    data = reshape(EMG_aligned_eye_rh(:, ch_idx, :), size(EMG_aligned_eye_rh, 1), size(EMG_aligned_eye_rh, 3));
    subplot(1, 2, 2)
    imagesc(time, 1:size(data, 1), data, [0, mean(mean(EMG_aligned_eye_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_eye_rh(:, ch_idx, :), 0, 3))]); % Plot matrix as image
    colormap('jet'); % Use 'jet' or any preferred colormap
    colorbar; % Add color scale
    xlabel('Time');
    ylabel('Trials');
    title('Trial Magnitude Over Time aligned to saccade onset');
    
    sgtitle(['Ch ', num2str(ch_idx)])
    saveas(gcf, ['Trial_EMG_color_plot_aligned_eye_ch' num2str(ch_idx) '.png']);

end

cd('..')
close all


%% EMG traces across trials
% cd('figures')
% EMG_aligned_tar_lh = EMG_aligned_tar_lh(1:end-1, :, :);
% EMG_aligned_eye_lh = EMG_aligned_eye_lh(1:end-1, :, :);

for ch_idx = 3
    
    figure
    subplot(1, 2, 1)
    analog_plot(-time_pre_tar:time_post_tar, reshape(EMG_aligned_tar_lh(:, ch_idx, :), size(EMG_aligned_tar_lh, 1), time_pre_tar+time_post_tar+1), 'Left', [0, 10000])
    xlabel('time (ms)')
    ylim([0, size(EMG_aligned_tar_lh, 1)*0.25+3])

    subplot(1, 2, 2)
    analog_plot(-time_pre_tar:time_post_tar, reshape(EMG_aligned_tar_rh(:, ch_idx, :), size(EMG_aligned_tar_rh, 1), time_pre_tar+time_post_tar+1), 'Right', [0, 10000])
    xlabel('time (ms)')
    ylim([0, size(EMG_aligned_tar_rh, 1)*0.25+3])
    sgtitle(['Aligned to target, ch ', num2str(ch_idx)])

%     saveas(gcf, ['Trial_EMG_traces_plot_aligned_tar_ch' num2str(ch_idx) '.png']);

end
% close all
for ch_idx = 3
    
    figure
    subplot(1, 2, 1)
    analog_plot(-time_pre_eye:time_post_eye, reshape(EMG_aligned_eye_lh(:, ch_idx, :), size(EMG_aligned_eye_lh, 1), time_pre_eye+time_post_eye+1), 'Left', [0, 10000])
    xlabel('time (ms)')
    ylim([0, size(EMG_aligned_eye_lh, 1)*0.25+3])
  
    subplot(1, 2, 2)
    analog_plot(-time_pre_eye:time_post_eye, reshape(EMG_aligned_eye_rh(:, ch_idx, :), size(EMG_aligned_eye_rh, 1), time_pre_eye+time_post_eye+1), 'Right', [0, 10000])
    xlabel('time (ms)')
    ylim([0, size(EMG_aligned_eye_rh, 1)*0.25+3])
    sgtitle(['Aligned to saccade, ch ', num2str(ch_idx)])

%     saveas(gcf, ['Trial_EMG_traces_plot_aligned_eye_ch' num2str(ch_idx) '.png']);

end
% 
% cd('..')
% close all
%% data aligned to target onset / single channel / average EMG + Eye position

ch_idx = 2;

subplot(3, 1, 1)
plot(-time_pre_tar:time_post_tar, mean(abs(reshape(EMG_aligned_tar_ru(:, ch_idx, :), size(EMG_aligned_tar_ru, 1), size(EMG_aligned_tar_ru, 3))), 1))
hold on
plot(-time_pre_tar:time_post_tar, mean(abs(reshape(EMG_aligned_tar_rd(:, ch_idx, :), size(EMG_aligned_tar_rd, 1), size(EMG_aligned_tar_rd, 3))), 1))
plot(-time_pre_tar:time_post_tar, mean(abs(reshape(EMG_aligned_tar_rh(:, ch_idx, :), size(EMG_aligned_tar_rh, 1), size(EMG_aligned_tar_rh, 3))), 1))
plot(-time_pre_tar:time_post_tar, mean(abs(reshape(EMG_aligned_tar_lu(:, ch_idx, :), size(EMG_aligned_tar_lu, 1), size(EMG_aligned_tar_lu, 3))), 1))
plot(-time_pre_tar:time_post_tar, mean(abs(reshape(EMG_aligned_tar_ld(:, ch_idx, :), size(EMG_aligned_tar_ld, 1), size(EMG_aligned_tar_ld, 3))), 1))
plot(-time_pre_tar:time_post_tar, mean(abs(reshape(EMG_aligned_tar_lh(:, ch_idx, :), size(EMG_aligned_tar_lh, 1), size(EMG_aligned_tar_lh, 3))), 1))

legend({'Right Up', 'Right Down', 'Right Hori', 'Left Up', 'Left Down', 'Left Hori'})
% legend({'Right Down', 'Right Hori', 'Left Down', 'Left Hori'})

title('Average abs EMG')
xlim([-time_pre_tar, time_post_tar])

subplot(3, 1, 2)
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_ru(:, 1, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
hold on
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_rd(:, 1, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_rh(:, 1, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_lu(:, 1, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_ld(:, 1, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_lh(:, 1, :), 1), 1, time_pre_tar+time_post_tar+1), 10))

legend({'Right Up', 'Right Down', 'Right Hori', 'Left Up', 'Left Down', 'Left Hori'})
title('Eye X')
xlim([-time_pre_tar, time_post_tar])

subplot(3, 1, 3)
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_ru(:, 2, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
hold on
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_rd(:, 2, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_rh(:, 2, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_lu(:, 2, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_ld(:, 2, :), 1), 1, time_pre_tar+time_post_tar+1), 10))
plot(-time_pre_tar:time_post_tar, movmean(reshape(mean(Eye_aligned_tar_lh(:, 2, :), 1), 1, time_pre_tar+time_post_tar+1), 10))

legend({'Right Up', 'Right Down', 'Right Hori', 'Left Up', 'Left Down', 'Left Hori'})
title('Eye Y')
xlim([-time_pre_tar, time_post_tar])

sgtitle(['Channel number: ',  num2str(ch_idx)])

%% data aligned to saccade onset / single channel / average EMG + Eye position

ch_idx = 7;

subplot(3, 1, 1)
plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_ru(:, ch_idx, :), size(EMG_aligned_eye_ru, 1), size(EMG_aligned_eye_ru, 3))), 1))
hold on
plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_rd(:, ch_idx, :), size(EMG_aligned_eye_rd, 1), size(EMG_aligned_eye_rd, 3))), 1))
plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_rh(:, ch_idx, :), size(EMG_aligned_eye_rh, 1), size(EMG_aligned_eye_rh, 3))), 1))
plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_lu(:, ch_idx, :), size(EMG_aligned_eye_lu, 1), size(EMG_aligned_eye_lu, 3))), 1))
plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_ld(:, ch_idx, :), size(EMG_aligned_eye_ld, 1), size(EMG_aligned_eye_ld, 3))), 1))
plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_lh(:, ch_idx, :), size(EMG_aligned_eye_lh, 1), size(EMG_aligned_eye_lh, 3))), 1))

legend({'Right Up', 'Right Down', 'Right Hori', 'Left Up', 'Left Down', 'Left Hori'})
% legend({'Right Down', 'Right Hori', 'Left Down', 'Left Hori'})

title('Average abs EMG')
xlim([-time_pre_eye, time_post_eye])

subplot(3, 1, 2)
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_ru(:, 1, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
hold on
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_rd(:, 1, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_rh(:, 1, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_lu(:, 1, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_ld(:, 1, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_lh(:, 1, :), 1), 1, time_pre_eye+time_post_eye+1), 10))

legend({'Right Up', 'Right Down', 'Right Hori', 'Left Up', 'Left Down', 'Left Hori'})
title('Eye X')
xlim([-time_pre_eye, time_post_eye])

subplot(3, 1, 3)
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_ru(:, 2, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
hold on
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_rd(:, 2, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_rh(:, 2, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_lu(:, 2, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_ld(:, 2, :), 1), 1, time_pre_eye+time_post_eye+1), 10))
plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(Eye_aligned_eye_lh(:, 2, :), 1), 1, time_pre_eye+time_post_eye+1), 10))

legend({'Right Up', 'Right Down', 'Right Hori', 'Left Up', 'Left Down', 'Left Hori'})
title('Eye Y')
xlim([-time_pre_eye, time_post_eye])

sgtitle(['Channel number: ',  num2str(ch_idx)])

%% data aligned to saccade onset / all channels / average EMG + Eye position

for ch_idx = 1:16

    if(ch_idx <= 8)
        subplot(8, 2, 2*ch_idx-1)
    else
        subplot(8, 2, 2*(ch_idx-8))
    end

    plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(abs(EMG_aligned_eye_ru(:, ch_idx, :)), 1), 1, time_pre_eye+time_post_eye+1), 10))
    hold on
    plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(abs(EMG_aligned_eye_rd(:, ch_idx, :)), 1), 1, time_pre_eye+time_post_eye+1), 10))
    plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(abs(EMG_aligned_eye_rh(:, ch_idx, :)), 1), 1, time_pre_eye+time_post_eye+1), 10))
    plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(abs(EMG_aligned_eye_lu(:, ch_idx, :)), 1), 1, time_pre_eye+time_post_eye+1), 10))
    plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(abs(EMG_aligned_eye_ld(:, ch_idx, :)), 1), 1, time_pre_eye+time_post_eye+1), 10))
    plot(-time_pre_eye:time_post_eye, movmean(reshape(mean(abs(EMG_aligned_eye_lh(:, ch_idx, :)), 1), 1, time_pre_eye+time_post_eye+1), 10))
    
    legend({'Right Up', 'Right Down', 'Right Hori', 'Left Up', 'Left Down', 'Left Hori'})
    title(['ch ', num2str(ch_idx)])
    xlim([-time_pre_eye, time_post_eye])
    ylim([0, 500])
end

%% Correlations 

% I put all trials concatenated together

% Original channel labels
channel_labels = {'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10', ...
                  'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16'};

% Remove unwanted channels
data = abs(cat(1, EMG_aligned_eye_ru, EMG_aligned_eye_rh, EMG_aligned_eye_rd, EMG_aligned_eye_lu, EMG_aligned_eye_lh, EMG_aligned_eye_ld));
channels_to_remove = [];
EMG_2d = reshape(permute(data, [2, 1, 3]), size(data, 2), []);
EMG_2d(channels_to_remove, :) = [];

% Update the channel labels
channel_labels(channels_to_remove) = [];

% Normalize the data
EMG_2d_norm = (EMG_2d - mean(EMG_2d, 2)) ./ std(EMG_2d, 0, 2);

% Compute correlation
rho = corr(EMG_2d_norm');

% Plot correlation matrix
imagesc(rho);
colormap('jet');
colorbar;

% Set correct labels
set(gca, 'YTick', 1:length(channel_labels));
set(gca, 'YTickLabel', channel_labels);
set(gca, 'XTick', 1:length(channel_labels));
set(gca, 'XTickLabel', channel_labels);
xtickangle(45); % Rotate x-axis labels for better readability
title('Correlation abs EMG aligned to saccade onset')

%%

st40 = find(rec_events{1}(1, :) == 40);


%% Threshold crossing spike detection

% Looking at PSTHs aligned to some event
EMG_clean = EMG;
spike_times = []; cluster = [];
for ch_idx = 1:16
    data = abs(EMG_clean(ch_idx, :));
    threshold = 0.5*std(EMG_clean(ch_idx, :));
    crossing = (data(1:end-1) < threshold) & (data(2:end) >= threshold);
    MU_spike_times = find(crossing) + 1;
    
%     sac_idx = find(output_clean(:, 10) > 5);
%     event_times_r = output_clean(sac_idx, 3);
%     sac_idx = find(output_clean(:, 10) < -5);
%     event_times_l = output_clean(sac_idx, 3);
    
    spike_times = [spike_times, MU_spike_times];
    cluster = [cluster, zeros(size(MU_spike_times)) + ch_idx];
end
% 
event_times_r = (PD1_Onsets(PD_rh)+Eye_RT(PD_rh))';
event_times_l = (PD1_Onsets(PD_lh)+Eye_RT(PD_lh))';

% event_times_r = (PD1_Onsets(PD_rh))';
% event_times_l = (PD1_Onsets(PD_lh))'; 

window = [-0.3, 0.3]; % look at spike times from 0.3 sec before each event to 1 sec after

% if your events come in different types, like different orientations of a
% visual stimulus, then you can provide those values as "trial groups",
% which will be used to construct a tuning curve.
event_times = [event_times_r', event_times_l'];

trialGroups = [zeros(size(event_times_r'))+2, zeros(size(event_times_l'))+3];

psthViewer(spike_times/1000, cluster, event_times/1000, window, trialGroups);

%%
% PSTHs across depth

depthBinSize = 1; % in units of the channel coordinates, in this case µm
timeBinSize = 0.005; % seconds
bslWin = [-0.5, -0.2]; % window in which to compute "baseline" rates for normalization
psthType = 'norm'; % show the normalized version
eventName = 'stimulus onset'; % for figure labeling

[timeBins, depthBins, allP, normVals] = psthByDepth(spike_times/1000, cluster, ...
    depthBinSize, timeBinSize, event_times_r/1000, window, bslWin);

[timeBins2, depthBins2, allP2, normVals2] = psthByDepth(spike_times/1000, cluster, ...
    depthBinSize, timeBinSize, event_times_l/1000, window, bslWin);

figure;
plotPSTHbyDepth(timeBins, depthBins, allP - allP2, eventName, psthType);
ylim([-0.5, 15.5])

%%

% data = abs(EMG_aligned_r(:, :, :));
% 
% channel_idx = 16; % Select the channel to plot
% time = 1:size(data, 3); % Time vector (1x2000)
% data = data(:, channel_idx, :);
% 
% % Compute mean and standard deviation across trials
% mean_signal = squeeze(mean(data, 1))'; % 1xT -> 1x2000, transpose to row vector
% std_signal = squeeze(std(data, [], 1))'; % 1xT -> 1x2000, transpose to row vector
% 
% % Define the shaded region (upper and lower bounds)
% upper_bound = mean_signal + std_signal;
% lower_bound = mean_signal - std_signal;
% 
% % Apply smoothing (choose one method)
% windowSize = 5; % Moving average window size
% smoothed_mean_r = movmean(mean_signal, windowSize);
% smoothed_upper = movmean(upper_bound, windowSize);
% smoothed_lower = movmean(lower_bound, windowSize);
% 
% % Create patch region correctly
% x_patch_r = [time, fliplr(time)];
% y_patch_r = [smoothed_upper, fliplr(smoothed_lower)];
% 
% data = abs(EMG_aligned_l(:, :, :));
% 
% time = -1000:1000; % Time vector (1x2000)
% data = data(:, channel_idx, :);
% 
% % Compute mean and standard deviation across trials
% mean_signal = squeeze(mean(data, 1))'; % 1xT -> 1x2000, transpose to row vector
% std_signal = squeeze(std(data, [], 1))'; % 1xT -> 1x2000, transpose to row vector
% 
% % Define the shaded region (upper and lower bounds)
% upper_bound = mean_signal + std_signal;
% lower_bound = mean_signal - std_signal;
% 
% % Apply smoothing (choose one method)
% windowSize = 5; % Moving average window size
% smoothed_mean_l = movmean(mean_signal, windowSize);
% smoothed_upper = movmean(upper_bound, windowSize);
% smoothed_lower = movmean(lower_bound, windowSize);
% 
% % Create patch region correctly
% x_patch_l = [time, fliplr(time)];
% y_patch_l = [smoothed_upper, fliplr(smoothed_lower)];
% 
% % Plot
% figure;
% hold on;
% plot(time, smoothed_mean_r, 'r', 'LineWidth', 1.5); % Smoothed mean signal
% % fill(x_patch_l, y_patch_l, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none'); % Shaded region
% 
% plot(time, smoothed_mean_l, 'b', 'LineWidth', 1.5); % Smoothed mean signal
% % fill(x_patch_r, y_patch_r, 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none'); % Shaded region
% legend({'Right', 'Left'})
% 
% xlabel('Time');
% ylabel('Signal');
% title(['Channel ', num2str(channel_idx)]);
% grid on;
% hold off;
% xlim([-100, 1000])

%%

function [x_calibrated, y_calibrated] = calibrateEyeRipple(EyeRPx_tr, EyeRPy_tr, EyeMLx_tr, EyeMLy_tr, EyeRPx, EyeRPy)

raw_signal_x = EyeRPx_tr';
calibrated_signal_x = EyeMLx_tr';
[c, lags] = xcorr(calibrated_signal_x, raw_signal_x);
[~, I] = max(c);
lag = lags(I);
lag = 0; % You need that for some trials *****
if lag > 0
    raw_signal_x_aligned = raw_signal_x(1:end-lag);
    calibrated_signal_x_aligned = calibrated_signal_x(lag+1:end);
elseif lag < 0
    lag2 = abs(lag);
    raw_signal_x_aligned = raw_signal_x(lag2+1:end);
    calibrated_signal_x_aligned = calibrated_signal_x(1:end-lag2);
else
    raw_signal_x_aligned = raw_signal_x;
    calibrated_signal_x_aligned = calibrated_signal_x;
end

min_length = min(length(raw_signal_x_aligned), length(calibrated_signal_x_aligned));
eyeX = raw_signal_x_aligned(1:min_length);
eyeX_ML = calibrated_signal_x_aligned(1:min_length);

raw_signal_x = EyeRPy_tr';
calibrated_signal_x = EyeMLy_tr';

if lag > 0
    raw_signal_x_aligned = raw_signal_x(1:end-lag);
    calibrated_signal_x_aligned = calibrated_signal_x(lag+1:end);
elseif lag < 0
    lag2 = abs(lag);
    raw_signal_x_aligned = raw_signal_x(lag2+1:end);
    calibrated_signal_x_aligned = calibrated_signal_x(1:end-lag2);
else
    raw_signal_x_aligned = raw_signal_x;
    calibrated_signal_x_aligned = calibrated_signal_x;
end

min_length = min(length(raw_signal_x_aligned), length(calibrated_signal_x_aligned));
eyeY = raw_signal_x_aligned(1:min_length);
eyeY_ML = calibrated_signal_x_aligned(1:min_length);

% Example raw and calibrated snippets
raw_snippet_x = eyeX'; % Replace with your actual snippet data
raw_snippet_y = eyeY'; % Replace with your actual snippet data
calibrated_snippet_x = eyeX_ML'; % Replace with your actual snippet data
calibrated_snippet_y = eyeY_ML'; % Replace with your actual snippet data

% Calculate the mean of the snippets
mean_raw_x = mean(raw_snippet_x);
mean_raw_y = mean(raw_snippet_y);
mean_calibrated_x = mean(calibrated_snippet_x);
mean_calibrated_y = mean(calibrated_snippet_y);

% Center the data around the mean
raw_snippet_x_centered = raw_snippet_x - mean_raw_x;
raw_snippet_y_centered = raw_snippet_y - mean_raw_y;
calibrated_snippet_x_centered = calibrated_snippet_x - mean_calibrated_x;
calibrated_snippet_y_centered = calibrated_snippet_y - mean_calibrated_y;

% Step 1: Calculate the gain and rotation matrix
A = [raw_snippet_x_centered; raw_snippet_y_centered];
B = [calibrated_snippet_x_centered; calibrated_snippet_y_centered];

% Solve the linear system to find rotation matrix R and scaling factor s
X = A' \ B';
R_s = X';

% Extract scaling factor and rotation matrix
s = sqrt(R_s(1,1)^2 + R_s(1,2)^2); % Scaling factor (gain)
R = R_s / s; % Rotation matrix

% Step 2: Apply the affine transformation to the entire raw signal
raw_signal_x = EyeRPx; % Replace with your actual entire raw signal
raw_signal_y = EyeRPy; % Replace with your actual entire raw signal

% Center the entire raw signal
raw_signal_x_centered = raw_signal_x - mean_raw_x;
raw_signal_y_centered = raw_signal_y - mean_raw_y;

% Apply the transformation
transformed_signal = R * [raw_signal_x_centered; raw_signal_y_centered] * s;

% Translate the transformed signal
x_calibrated = transformed_signal(1, :) + mean_calibrated_x;
y_calibrated = transformed_signal(2, :) + mean_calibrated_y;

save('Calibration', 'R', 's', 'mean_raw_x', 'mean_raw_y', 'mean_calibrated_x', 'mean_calibrated_y')
end