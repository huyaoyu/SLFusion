function [fig] = plot_3_channel(baseDir, baseFn, winTitle)
% fig: The handle of the figure.
% baseDir: The name of the base directory, string.
% baseFn: The base filename, string.
% winTitle: The window title, string.

% Populate a cell.
cCell = cell(3, 1);

for I = 1:1:3
    fn = sprintf('%s/%s_%d.dat', baseDir, baseFn, I - 1);
    cCell{I, 1} = load(fn);
end

% Show the figure.
fig = merge_plot_channels(cCell, [3, 2, 1], winTitle);
