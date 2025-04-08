clc; clear;

folder_path = 'C:\Users\CorneilLab\Desktop\MayoTools';
addpath(genpath(folder_path));

folder_path = 'C:\Users\CorneilLab\Desktop\SortingTools';
addpath(genpath(folder_path));

folder_path = 'C:\Users\CorneilLab\Desktop\SharedUtils';
addpath(genpath(folder_path));

folder_path = 'C:\Users\CorneilLab\Documents\NIMH_MonkeyLogic_2.2';
addpath(genpath(folder_path));

sesName =                   '2025-03-11_12-51-59';
animalName =                'Trex';

Folders.rec_path = ['C:\Users\CorneilLab\Desktop\Myo_Data\Raw_data\', animalName,'\', sesName, '\'];
cd(Folders.rec_path)

load('rec_Analog_2.mat')
load('rec_EMG_1k_2.mat')
load('Calibration.mat')

sampling_rate = 30000;

Analog_Data_mV = double(rec_Analog{1, 1}.data);

EyeRPx = double(Analog_Data_mV(1, :));
EyeRPy = double(Analog_Data_mV(2, :));

% Calibrate ripple data
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


%% Blink Detection

[blink_onsets, blink_offsets] = detectBlink(x_calibrated', y_calibrated');
blinks = find(blink_offsets - blink_onsets > 30);     % These are considered as blinks --> I put nan there
nonblinks = find(blink_offsets - blink_onsets <= 30); % These are short-period spikes in eye signal --> I interpolate them
% Plot to visualize the blinks
signal = [abs(x_calibrated)];
figure;
plot(signal, 'b'); hold on;
plot(blink_onsets(blinks), signal(blink_onsets(blinks)), 'ro', 'MarkerSize', 8, 'DisplayName', 'Blink Onset');
plot(blink_offsets(blinks), signal(blink_offsets(blinks)), 'go', 'MarkerSize', 8, 'DisplayName', 'Blink Offset');
plot(blink_onsets(nonblinks), signal(blink_onsets(nonblinks)), 'ko', 'MarkerSize', 8, 'DisplayName', 'Blink Onset');
plot(blink_offsets(nonblinks), signal(blink_offsets(nonblinks)), 'bo', 'MarkerSize', 8, 'DisplayName', 'Blink Offset');
title('Eye Tracking Signal with Blink Events');
legend;
x_cleaned = x_calibrated;
% Replace blink periods with NaN
for i = 1:length(blink_onsets(blinks))
    onset = blink_onsets(blinks(i));
    offset = blink_offsets(blinks(i));

    x_cleaned(onset:offset) = NaN;  % Set the blink period to NaN
end
y_cleaned = y_calibrated;
% Replace blink periods with NaN
for i = 1:length(blink_onsets(blinks))
    onset = blink_onsets(blinks(i));
    offset = blink_offsets(blinks(i));
    y_cleaned(onset:offset) = NaN;  % Set the blink period to NaN
end
% Loop through each spike period and interpolate the signal
for i = 1:length(blink_onsets(nonblinks))
    % Find indices corresponding to the blink period
    blink_indices = blink_onsets(nonblinks(i)):blink_offsets(nonblinks(i));
    % Find the indices just before and after the blink period for interpolation
    start_index = blink_onsets(nonblinks(i))-2;
    end_index = blink_offsets(nonblinks(i))+2;
    % Perform linear interpolation between the start and end of the blink period
    x_cleaned(blink_indices) = interp1([start_index, end_index], ...
                               [x_cleaned(start_index), x_cleaned(end_index)], ...
                               blink_indices, 'linear');
    % Perform linear interpolation between the start and end of the blink period
    y_cleaned(blink_indices) = interp1([start_index, end_index], ...
                               [y_cleaned(start_index), y_cleaned(end_index)], ...
                               blink_indices, 'linear');
end
% Plot the original and NaN-modified signals
figure;
plot(x_cleaned, 'b');
% Add patches for blink periods
for i = 1:length(blink_onsets(blinks))
    % Define the x and y coordinates for the patch
    patch_x = [blink_onsets((blinks(i))), blink_offsets((blinks(i))), blink_offsets((blinks(i))), blink_onsets((blinks(i)))];
    patch_y = [min(x_cleaned), min(x_cleaned), max(x_cleaned), max(x_cleaned)];
    % Create the patch (shaded area)
    patch(patch_x, patch_y, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end
% Add patches for blink periods
for i = 1:length(blink_onsets(nonblinks))
    % Define the x and y coordinates for the patch
    patch_x = [blink_onsets((nonblinks(i))), blink_offsets((nonblinks(i))), blink_offsets((nonblinks(i))), blink_onsets((nonblinks(i)))];
    patch_y = [min(x_cleaned), min(x_cleaned), max(x_cleaned), max(x_cleaned)];
    % Create the patch (shaded area)
    patch(patch_x, patch_y, 'k', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end
title('Eye Tracking Signal with Blink Periods Replaced by NaN');
legend('Signal with Blinks as NaN');
xlabel('Sample');
ylabel('Signal');
% Apply median filtering
window_size = 100;  % Choose an appropriate window size
x_cleaned_filt = medfilt1(x_cleaned, window_size, 'includenan');
y_cleaned_filt = medfilt1(y_cleaned, window_size, 'includenan');
% Plot the original and NaN-modified signals
figure;
plot(x_cleaned, 'b'); hold on;
plot(x_cleaned_filt, 'r');
% Add patches for blink periods
for i = 1:length(blink_onsets(blinks))
    % Define the x and y coordinates for the patch
    patch_x = [blink_onsets((blinks(i))), blink_offsets((blinks(i))), blink_offsets((blinks(i))), blink_onsets((blinks(i)))];
    patch_y = [min(x_cleaned), min(x_cleaned), max(x_cleaned), max(x_cleaned)];
    % Create the patch (shaded area)
    patch(patch_x, patch_y, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end
% Add patches for blink periods
for i = 1:length(blink_onsets(nonblinks))
    % Define the x and y coordinates for the patch
    patch_x = [blink_onsets((nonblinks(i))), blink_offsets((nonblinks(i))), blink_offsets((nonblinks(i))), blink_onsets((nonblinks(i)))];
    patch_y = [min(x_cleaned), min(x_cleaned), max(x_cleaned), max(x_cleaned)];
    % Create the patch (shaded area)
    patch(patch_x, patch_y, 'k', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
end
title('Filtered: Red;     Unfiltered: Blue')
%%
% Saccade detection algorithm needs continous data so I interpolate now and
% then remove saccades during blinks
% % Interpolate to fill the gaps
x_interp = interp1(1:length(x_cleaned_filt), x_cleaned_filt, 1:length(x_cleaned_filt), 'spline');
y_interp = interp1(1:length(y_cleaned_filt), y_cleaned_filt, 1:length(y_cleaned_filt), 'spline');

plot(x_cleaned_filt, 'r'); hold on;
plot(y_interp)

%%

x = x_interp; % Example x data (screen width in pixels)
y = y_interp; % Example y data (screen height in pixels)

% Define grid size and range
grid_size = [20, 20]; % 5x5 grid
x_range = [-40, 40]; % Horizontal range
y_range = [-40, 40]; % Vertical range

% Create grid edges
x_edges = linspace(x_range(1), x_range(2), grid_size(1) + 1);
y_edges = linspace(y_range(1), y_range(2), grid_size(2) + 1);

% Compute fixation density using histcounts2
fixation_density = histcounts2(x, y, x_edges, y_edges);

% Plot using pcolor for correct alignment
figure;
pcolor(x_edges, y_edges, [fixation_density', zeros(size(fixation_density, 1), 1); zeros(1, size(fixation_density, 2) + 1)]);
shading flat; % Remove gridlines between cells
colormap('hot');
colorbar;
xlabel('Horizontal Position (deg)');
ylabel('Vertical Position (deg)');
title('Fixation Heatmap');

% Add grid lines for better visualization
hold on;
for i = 1:length(x_edges)
    xline(x_edges(i), 'k');
end
for i = 1:length(y_edges)
    yline(y_edges(i), 'k');
end
hold off;

% Set axis limits and labels to full range
set(gca, 'XTick', x_edges);
set(gca, 'XTickLabel', round(x_edges, 1));
set(gca, 'YTick', y_edges);
set(gca, 'YTickLabel', round(y_edges, 1));

% Invert y-axis to match Cartesian coordinates
set(gca, 'YDir', 'normal');

figure
subplot(1, 2, 1)
histogram(x_interp)
title('X')
subplot(1, 2, 2)
histogram(y_interp)
title('Y')

%%

% Let's remove the reward artifact first.
% load('rec_events.mat')
% load('rec_Z.mat')

EMG = rec_EMG_1k{1, 1}.data(1:16, :);

% figure
time = (0:length(EMG(1, :))-1) / 1000;
% plot(time, EMG(1, :)); hold on
% plot(time(art1), EMG(1, art1), 'o')
% rew = find(rec_events{1, 1}(1, :) == 62);
% xline(rec_events{1, 1}(2, rew) - rec_events{1, 1}(2, 1) + time(8864) - (rec_events{1, 1}(2, rew(2)) - rec_events{1, 1}(2, 1)))

figure
threshold = 1000;
artifact_periods = detect_artifacts(EMG, threshold, 5);
plot(time, EMG(14, :)); hold on
xline(time(artifact_periods(:, 1)), 'r');
xline(time(artifact_periods(:, 2)), 'b');

yline(threshold)

EMG_clean = EMG;
for i = 1:size(artifact_periods, 1)
    EMG_clean(:, artifact_periods(i, 1)-100:artifact_periods(i, 2)+100) = zeros(16, length(artifact_periods(i, 1)-100:artifact_periods(i, 2)+100));
end

%%

offset = 0; % Initial vertical offset
vertical_shift = -1.1; % Amount to shift each plot vertically
figure; 
subplot(1, 2, 1); hold on;
for kk = 1:16

    data = EMG(kk, :); % Use the first channel of data

    % Plot with vertical shift
    plot(normalize(data - mean(data),'range', [0, 1])+offset, 'LineWidth', 1.5);

    % Update the offset for the next stream
    offset = offset + vertical_shift;
end
title('EMG + reward artifact')

subplot(1, 2, 2); hold on;
for kk = 1:16

    data = EMG_clean(kk, :); % Use the first channel of data

    % Plot with vertical shift
    plot(normalize(data - mean(data),'range', [0, 1])+offset, 'LineWidth', 1.5);

    % Update the offset for the next stream
    offset = offset + vertical_shift;
end
title('Clean EMG')


%%
% now find saccade onsets
% x_interp, y_interp
Eh = x_interp'; % Horizontal Eye Position
Ev = y_interp'; % Vertical Eye Position
dEh = diff(Eh')*1000;
dEv = diff(Ev')*1000;

output = sacdetector(dEh,dEv,Eh',Ev',[100, 100, 50],0);

onsets = output(:, 3);
min_distance = 100;

idx = [1];
clean_onsets = onsets(1); % Keep the first onset
for i = 2:length(onsets)
    if onsets(i) - clean_onsets(end) > min_distance
        clean_onsets = [clean_onsets; onsets(i)];
        idx = [idx, i];
    end
end

output_clean = output(idx, :);

plot(x_interp(1:10000))
xline(output_clean(output_clean(:, 3) > 0 & output_clean(:, 3) < 10000, 3))

%%
time_pre_eye = 100;
time_post_eye = 100;

ch_idx = 2;

sac_idx = find(output_clean(:, 10) > 5);
EMG_aligned_eye_rh = zeros(length(sac_idx), 16, time_pre_eye+time_post_eye+1);
for i = 1:length(sac_idx)
    EMG_aligned_eye_rh(i, :, :) = EMG_clean(:, output_clean(sac_idx(i), 3)-time_pre_eye:output_clean(sac_idx(i), 3)+time_post_eye);
end

figure
subplot(1, 2, 2)
k = 15;
plot(-time_pre_eye:time_post_eye, abs(reshape(EMG_aligned_eye_rh(:, ch_idx, :), size(EMG_aligned_eye_rh, 1), size(EMG_aligned_eye_rh, 3))'), 'k')
hold on
plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_rh(:, ch_idx, :), size(EMG_aligned_eye_rh, 1), size(EMG_aligned_eye_rh, 3))), 1), 'r')
ylim([0, mean(mean(EMG_aligned_eye_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_eye_rh(:, ch_idx, :), 0, 3))])
xlim([-time_pre_eye, time_post_eye])
title('Right')

sac_idx = find(output_clean(:, 10) < -5);
EMG_aligned_eye_lh = zeros(length(sac_idx), 16, time_pre_eye+time_post_eye+1);
for i = 1:length(sac_idx)
    EMG_aligned_eye_lh(i, :, :) = EMG_clean(:, output_clean(sac_idx(i), 3)-time_pre_eye:output_clean(sac_idx(i), 3)+time_post_eye);
end
subplot(1, 2, 1)
k = 30;
plot(-time_pre_eye:time_post_eye, abs(reshape(EMG_aligned_eye_lh(:, ch_idx, :), size(EMG_aligned_eye_lh, 1), size(EMG_aligned_eye_lh, 3))'), 'k')
hold on
plot(-time_pre_eye:time_post_eye, mean(abs(reshape(EMG_aligned_eye_lh(:, ch_idx, :), size(EMG_aligned_eye_lh, 1), size(EMG_aligned_eye_lh, 3))), 1), 'r')
ylim([0, mean(mean(EMG_aligned_eye_rh(:, ch_idx, :))) + k*mean(std(EMG_aligned_eye_rh(:, ch_idx, :), 0, 3))])
xlim([-time_pre_eye, time_post_eye])
title('Left')

figure

subplot(1, 2, 1)
analog_plot(-time_pre_eye:time_post_eye, reshape(EMG_aligned_eye_lh(:, ch_idx, :), size(EMG_aligned_eye_lh, 1), time_pre_eye+time_post_eye+1), 'Left', [0, 4000])
xlabel('time (ms)')
ylim([0, size(EMG_aligned_eye_lh, 1)*0.25+3])

subplot(1, 2, 2)
analog_plot(-time_pre_eye:time_post_eye, reshape(EMG_aligned_eye_rh(:, ch_idx, :), size(EMG_aligned_eye_rh, 1), time_pre_eye+time_post_eye+1), 'Right', [0, 4000])
xlabel('time (ms)')
ylim([0, size(EMG_aligned_eye_lh, 1)*0.25+3])

%% Correlations 

% I put all trials concatenated together

% Original channel labels
channel_labels = {'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5', 'Ch6', 'Ch7', 'Ch8', 'Ch9', 'Ch10', ...
                  'Ch11', 'Ch12', 'Ch13', 'Ch14', 'Ch15', 'Ch16'};

% Remove unwanted channels
data = abs(cat(1, EMG_aligned_eye_rh, EMG_aligned_eye_lh));
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
figure
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

%% Threshold crossing spike detection

% Looking at PSTHs aligned to some event

spike_times = []; cluster = [];
ch_arr = 1:16;
ch_arr = ch_arr(~ismember(ch_arr, channels_to_remove));

for ch_idx = ch_arr
    data = EMG_clean(ch_idx, :);
    threshold = 3*std(EMG_clean(ch_idx, :));
    crossing = (data(1:end-1) < threshold) & (data(2:end) >= threshold);
    MU_spike_times = find(crossing) + 1;
    
    sac_idx = find(output_clean(:, 10) > 5);
    event_times_r = output_clean(sac_idx, 3);
    sac_idx = find(output_clean(:, 10) < -5);
    event_times_l = output_clean(sac_idx, 3);

    spike_times = [spike_times, MU_spike_times];
    cluster = [cluster, zeros(size(MU_spike_times)) + ch_idx];
end

window = [-0.5, 0.5]; % look at spike times from 0.3 sec before each event to 1 sec after

% if your events come in different types, like different orientations of a
% visual stimulus, then you can provide those values as "trial groups",
% which will be used to construct a tuning curve. Here we just give a
% vector of all ones.
event_times = [event_times_r', event_times_l'];

trialGroups = [zeros(size(event_times_r'))+2, zeros(size(event_times_l'))+3];

psthViewer(spike_times/1000, cluster, event_times/1000, window, trialGroups);

%%
% PSTHs across depth

depthBinSize = 1; % in units of the channel coordinates, in this case µm
timeBinSize = 0.005; % seconds
bslWin = [-0.5, -0.2]; % window in which to compute "baseline" rates for normalization
psthType = 'norm'; % show the normalized version
eventName = 'saccade onset'; % for figure labeling

[timeBins, depthBins, allP, normVals] = psthByDepth(spike_times/1000, cluster, ...
    depthBinSize, timeBinSize, event_times_r/1000, window, bslWin);

[timeBins2, depthBins2, allP2, normVals2] = psthByDepth(spike_times/1000, cluster, ...
    depthBinSize, timeBinSize, event_times_l/1000, window, bslWin);

figure;
plotPSTHbyDepth(timeBins, depthBins, allP - allP2, eventName, psthType);
ylim([-0.5, 15.5])
set(gca, 'YDir', 'reverse')

%%

function artifact_periods = detect_artifacts(data, threshold, min_channels)
    % data: 16 x time matrix
    % threshold: value to detect artifacts (e.g., 2400)
    % min_channels: minimum number of channels exceeding the threshold to consider it an artifact
    
    high_value_points = sum(data > threshold, 1) >= min_channels;
    
    % Find the start and end of artifact periods
    diff_points = diff([0, high_value_points, 0]); % Add padding to detect edges
    starts = find(diff_points == 1);
    ends = find(diff_points == -1) - 1;
    
    % Combine start and end times into a matrix
    artifact_periods = [starts', ends'];
end



function [blink_onsets, blink_offsets] = detectBlink(calibrated_eye_x_filt1, calibrated_eye_y_filt1)
    % Example data (replace this with your actual data)
    signal = [abs(calibrated_eye_x_filt1)];  % The eye-tracking signal (either x or y)
    % Example data (replace this with your actual data)
    threshold = 50;  % Define your blink threshold
    min_duration_below_threshold = 3;  % Minimum number of consecutive points below threshold to confirm offset
    % Find indices where signal exceeds the threshold
    above_threshold = signal > threshold;
    % Find transitions (onsets and tentative offsets)
    diff_signal = diff(above_threshold);
    blink_onsets = find(diff_signal == 1)-1;  % Onset: transition from below to above threshold
    tentative_offsets = find(diff_signal == -1) + 4; % Tentative Offset: transition from above to below threshold
    % Initialize blink offsets
    blink_offsets = tentative_offsets;
    % % Refine the detection of offsets
    % for i = 1:length(tentative_offsets)
    %     idx = tentative_offsets(i);
    %
    %     % Check if the signal stays below the threshold for at least min_duration_below_threshold points
    %     if all(signal(idx: min(idx + min_duration_below_threshold - 1, length(signal))) < threshold)
    %         blink_offsets = [blink_offsets; idx];  % Confirmed offset
    %     end
    % end
    % Handle cases where the signal starts or ends above the threshold
    if above_threshold(1)
        blink_onsets = [1; blink_onsets];  % Add the start of the signal as an onset
    end
    if above_threshold(end)
        blink_offsets = [blink_offsets; length(signal)];  % Add the end of the signal as an offset
    end
    % Sometimes there are multiple blink onsets! but just one offset
    % blink_onsets((find(diff(blink_onsets) <= 50))) = [];
end
