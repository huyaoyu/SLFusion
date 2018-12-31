function [fig] = merge_plot_channels(cCell, idx, winTitle)
% cCell: A cell contains all the channels. cCell is arranged in column.
% idx: An array contains the indices of cCell in RGB order.
% winTitle: The window title, string.
% fig: The newly created figure handle.

% Size.
[row, col] = size( cCell{1, 1} );

% Coordinates.
coorRow = 0:1:row;
coorCol = 0:1:col;

[x, y] = meshgrid( coorCol, coorRow );
z      = zeros( row + 1, col + 1 );
y      = y * -1;

% Number of channels.
nc = size(cCell, 1);

if ( 3 == nc )
    C = zeros(row, col, 3);
    C(:, :, 1) = cCell{idx(1), 1} / 255;
    C(:, :, 2) = cCell{idx(2), 1} / 255;
    C(:, :, 3) = cCell{idx(3), 1} / 255;
    
    fig = figure('Name', winTitle, 'NumberTitle', 'off');
    surf(x, y, z, C, 'FaceColor', 'flat');
elseif ( 1 == nc )
    fprintf('Error: Not implemented yet.\n');
    fig = 0;
else
    % Error.
    fprintf('Error: Only supports 1-channel or 3-channel input cell. nc = %d.\n', nc);
    fig = 0;
end

xlabel('x');
ylabel('y');
zlabel('z');
view(2);
