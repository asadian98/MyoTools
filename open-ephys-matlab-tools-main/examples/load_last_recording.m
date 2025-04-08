clc; clear;

folder_path = 'C:\Users\CorneilLab\Desktop\MayoTools';
addpath(genpath(folder_path));
    
% Define path to the recording
rec_path = 'C:\Users\CorneilLab\Desktop\Mayo\Raw_data\test20241213';

% Create a session (loads all data from the most recent recording)
session = Session(rec_path);

% Get the number of record nodes for this session
nRecordNodes = length(session.recordNodes);

% Iterate over the record nodes to access data
for i = 1:nRecordNodes

    node = session.recordNodes{i};

    for j = 1:length(node.recordings)

        % 1. Get the first recording 
        recording = node.recordings{1,j};

        % 2. Iterate over all continuous data streams in the recording 
        streamNames = recording.continuous.keys();

        for k = 1:length(streamNames)

            streamName = streamNames{k};
            disp(streamName)

            % Get the continuous data from the current stream
            continuousdata = recording.continuous(streamName);
            rec_continuousdata.data = continuousdata.samples;
            rec_continuousdata.timestamps = continuousdata.timestamps; 

            % Plot first channel of continuous data 
            if 1 
                plot(continuousdata.timestamps, continuousdata.samples(1,:), 'LineWidth', 1.5);
                title(recording.format, recording.format); hold on;
            end
            
            % 3. Overlay all available event data
            eventProcessors = recording.ttlEvents.keys();
            processor = eventProcessors{1};
            events = recording.ttlEvents(processor);
            
            % Just get valid events: 
            [C,ia,ic] = unique([double(events.full_words(events.state)), events.timestamp(events.state)], 'rows', 'stable');
            rec_events = C';
             
            if ~isempty(events)
                for n = 1:size(rec_events, 2)
                    line([rec_events(2, n), rec_events(2, n)], [-4000,2000], 'Color', 'red', 'LineWidth', 0.2);
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
    

% Save result 
% exportgraphics(gcf(), fullfile("examples", "load_last_recording.pdf"));