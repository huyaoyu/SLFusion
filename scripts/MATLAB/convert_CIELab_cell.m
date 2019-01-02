function [cCell2] = convert_CIELab_cell(cCell1)
% cCell1: The input cell of channels.
% cCell2: The output cell of channels.
% The number of channels must be three and be in a column.

% Get the dimensions of the channel.
[row, col] = size( cCell1{1, 1} );

% Stack the channels together.
img = zeros( row, col, 3 );

for I = 1:1:3
    img( :, :, I ) = cCell1{I, 1};
end % I

% Convert the data type into double precision floating point.
img = double( img );

% Shift the L*a*b values back to there normal ranges.
img(:, :, 1) = img(:, : ,1) * 100 / 255;
img(:, :, 2) = img(:, :, 2) - 128;
img(:, :, 3) = img(:, :, 3) - 128;

% Convert color space.
img = lab2rgb( img, 'OutputType','uint8', 'ColorSpace', 'adobe-rgb-1998' );

% Split the channels again.
cCell2 = cell(3, 1);

for I = 1:1:3
    cCell2{I, 1} = img(:, :, I);
end
