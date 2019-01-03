function add_rectangle_3d(fig, x, y, z, width, height, rgb)
% fig: The handle to the figure.
% x, y, z: The coordinates.
% width: The Length along x.
% height: The Length along y.
% rgb: RGB color, a three-element array. Must be a column vector.
%
% Note that this function assumes that the figure is flat and parallel to
% the xy plane. The z coordinate is the location of the intersection
% between the plane and the z-axis.
%

% Make the fig the current figure.
set(0, 'currentfigure', fig);

% Prepare the coordinates.
rx = [ x; x + width; x + width;  x ];
ry = [ y; y;         y + height; y + height ];
rz = [ z; z;         z;          z ];

% Plot the rectangle.
hold on;
fill3( rx, ry, rz, rgb );
hold off;