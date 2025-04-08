function analog_plot(Ttime,datain,plottitle,scale,marks1,marks2)
% Generic plotting function for any analog data.
%   Ttime: time scale. Must match the columns of Data
%   Data: each row is a different trial
%   title: title of subplot
%   scale: min and max range of dataset
%   marks1: optional set of marks, relative to 0 of Ttime
%   marks2: optional set of marks, relative to 0 of Ttime

min_data = scale(1);
max_data = scale(2);
range_data = (max_data - min_data) * .1;
datain_norm = datain / range_data;

hold on;
title(plottitle)
for i = 1:size(datain_norm,1)
    plot(Ttime,datain_norm(i,:)+i*.25,'k-')
    if nargin == 6
        % PLOT MARKS 1 and MARKS 2
        % PLOT MARKS 1
        if ~isnan(marks1(i))
            plot(marks1(i),i*.25,'ro','MarkerFaceColor','r');
        end
        if ~isnan(marks2(i))
            plot(marks2(i),i*.25,'g.');
        end
    elseif nargin == 5
        % PLOT MARKS 1
        if ~isnan(marks1(i))
            plot(marks1(i),i*.25,'ro','MarkerFaceColor','r');
        end
    end
end
line([0 0],[0 i*.25+1]);
axis([min(Ttime) max(Ttime) 0 i*.25+1])
