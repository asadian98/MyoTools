clc; clear;

folder_path = 'C:\Users\CorneilLab\Desktop\MyoTools';
addpath(genpath(folder_path));
    
folder_path = 'C:\Users\CorneilLab\Desktop\SortingTools';
addpath(genpath(folder_path));

folder_path = 'C:\Users\CorneilLab\Desktop\SharedUtils';
addpath(genpath(folder_path));

folder_path = 'C:\Users\CorneilLab\Documents\NIMH_MonkeyLogic_2.2';
addpath(genpath(folder_path)); 

% Define path to the recording
rec_path = 'C:\Users\CorneilLab\Desktop\Myo_Data\Raw_Data\Trex\Tomomichi_recording_myo';

% Create a session (loads all data from the most recent recording)
session = Session(rec_path);

% Get the number of record nodes for this session
nRecordNodes = length(session.recordNodes);

% Iterate over the record nodes to access data
for i = 1:nRecordNodes

    node = session.recordNodes{i};

    for j = 1:length(node.recordings)

        figure;
        hold on;
        offset = 0; % Initial vertical offset
        vertical_shift = 1.1; % Amount to shift each plot vertically

        % 1. Get the first recording 
        recording = node.recordings{1,j};

        % 2. Iterate over all continuous data streams in the recording 
        streamNames = recording.continuous.keys();

        for k = 1:length(streamNames)

            streamName = streamNames{k};
            disp(streamName)

            % Get the continuous data from the current stream
            continuousdata = recording.continuous(streamName);
            rec_continuousdata{j}.data = continuousdata.samples;
            rec_continuousdata{j}.timestamps = continuousdata.timestamps; 

            for kk = 1:24

                data = double(continuousdata.samples(kk, :)); % Use the first channel of data
    
                % Plot with vertical shift
                plot(continuousdata.timestamps, normalize(data - mean(data),'range', [0, 1])+offset, 'LineWidth', 1.5);
                
                % Add label for each data stream
                text(continuousdata.timestamps(1), offset, ['ADC', num2str(kk)], 'VerticalAlignment', 'bottom', 'FontSize', 8);
    
                % Update the offset for the next stream
                offset = offset + vertical_shift;
            end

            % 3. Overlay all available event data
            eventProcessors = recording.ttlEvents.keys();
            processor = eventProcessors{1};
            events = recording.ttlEvents(processor);
            
            % Just get valid events: 
            events_without_eye = events.line(find(events.line ~= 8 & events.line ~= -8 & events.line ~= -7));
            timestamp_without_eye = events.timestamp(find(events.line ~= 8 & events.line ~= -8 & events.line ~= -7));
            [rec_events{j}(1, :), rec_events{j}(2, :)] = getEvents_from_train_with_strobe(events_without_eye, timestamp_without_eye);
             
            if ~isempty(events)
                for n = 1:size(rec_events{j}, 2)
                    line([rec_events{j}(2, n), rec_events{j}(2, n)], [4.5,5.5], 'Color', 'red', 'LineWidth', 0.2);
                end
            end
            
        end
        
        % 5. Print any message events
        if recording.messages.Count > 0
            disp("Found Message Center events!");
            % recording.messages('MessageCenter');
        end

    end
end

%%

%% Load data
load Digastric_EMG_data_211015_163233.mat
sampling_rate=frequency_parameters.board_adc_sample_rate;

%% Set frequency limits for bandpass.
% In most cases, waveforms from isolated single units have frequency
% content between 500-2,000 Hz
filter_limits=[300 7000];

%% Choose which channels to plot.
% In this recording, channels 1-16 were recorded from the left digastric
% muscle, and channels 17-32 from the right digastric.
channel_vec=1:32;

%% Plotting

% Vertical separation between plotted channel traces
vertical_offset=3500; % units = microvolts

figure(1);clf;hold on

for xx=1:length(channel_vec)

    % Select channel number
    x=channel_vec(xx);

    % Bandpass each channel
    dat_filt(x,:) = bandpass(amplifier_data(x,:),filter_limits,sampling_rate);

    % Plot each channel
    plot([1:length(dat_filt(xx,:))]/sampling_rate,dat_filt(xx,:)+vertical_offset*xx);

    % Label each channel's data
    text(length(dat_filt(xx,:))/sampling_rate,+vertical_offset*xx,['Ch. ' num2str(x)]);
end

xlabel('Time (sec)');ylabel('Voltage ({\mu}V)')


%%

function [decoded_values, decoded_timestamps] = getEvents_from_train(train, timestamps)

    % Initialize the binary state for 6 digits (all start from 0)
    state = zeros(1, 6);
    
    % Initialize output storage
    decoded_values = [];
    decoded_timestamps = [];
    
    % Group events by unique timestamps
    [unique_timestamps, ~, group_indices] = unique(timestamps);
    
    % Process each group of events at the same timestamp
    for i = 1:length(unique_timestamps)
        % Find the indices of the events that occurred at this timestamp
        group_indices_at_time = find(group_indices == i);
        
        % Toggle the states of all digits in this group
        for idx = group_indices_at_time
            digit_index = train(idx);
            state(digit_index) = ~state(digit_index); % Flip 0 to 1 or 1 to 0
        end
        
        % Calculate the decimal value from the binary state
        decimal_value = sum(state .* 2.^(0:5)); % 6 digits, most significant bit is on the left
        
        % Store the decoded value and the timestamp
        decoded_values = [decoded_values, decimal_value];
        decoded_timestamps = [decoded_timestamps, unique_timestamps(i)];
    end

end

function [decoded_values, decoded_timestamps] = getEvents_from_train_with_strobe(train, timestamps)

    % Initialize the binary state for 6 digits and strobe (7th digit)
    state = zeros(1, 7);
    
    % Unique timestamps and their indices
    [unique_timestamps, ~, idx] = unique(timestamps);
    
    % Initialize output storage
    decoded_values = [];
    decoded_timestamps = [];
    
    % Process each unique timestamp
    for t = 1:length(unique_timestamps)
        % Get the indices of the train for this timestamp
        current_indices = find(idx == t);
        
        % Update the state for all digits at this timestamp
        for i = current_indices
            digit_index = abs(train(i)); % Get the absolute digit index
            state(digit_index) = train(i) > 0; % Set to 1 if positive, 0 if negative
        end
        
        % Check if strobe (digit 7) is up (1)
        if state(7) == 1
            % Calculate the decimal value of digits 1 to 6
            decimal_value = sum(state(1:6) .* 2.^(0:5)); % 6 digits, MSB is on the left
            
            % Store the timestamp of the strobe being up and the decoded value
            decoded_values = [decoded_values, decimal_value];
            decoded_timestamps = [decoded_timestamps, unique_timestamps(t)];
        end
    end
    
    % Because I'm using the strobe code, "decoded values" gives me two values (one when
    % the digits turn up and one when the strobe bit goes high)
    if(decoded_values(1) == decoded_values(2))
        decoded_values = [decoded_values(1), decoded_values(2:2:end)];
        decoded_timestamps = [decoded_timestamps(1), decoded_timestamps(2:2:end)];
    else
        decoded_values = decoded_values(1:2:end);
        decoded_timestamps = decoded_timestamps(1:2:end);
    end

end