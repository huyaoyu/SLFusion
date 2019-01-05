clc;
clear;
close all;

BASE_DIR   = '../../data/SLFusion/match_single_line_05_cost';
IMG_FN     = { 'img0.jpg'; 'img1.jpg' };
% CENTER_LOC = {
%     [3573, 1505];
%     [3573 - 600, 1505];
% };

CENTER_LOC = {
    [3035, 1505];
    [3035 - 600, 1505];
};

nFiles = size(IMG_FN, 1);

for I = 1:1:nFiles
    % Compose the filename.
    FN = sprintf('%s/%s', BASE_DIR, IMG_FN{I, 1});

    % Read the image.
    h   = figure('Name', IMG_FN{I, 1}, 'NumberTitle', 'off');
    img = imread(FN);

    imshow(img);

    hold on;
    axis on;

    % The cetner.
    x = CENTER_LOC{I, 1}(1);
    y = CENTER_LOC{I, 1}(2);
    
    % Plot a rectangle.
    rectangle(h.Children, 'Position',[x - 20, y - 20, 39, 39],...
      'EdgeColor', 'r',...
      'LineWidth', 1,...
      'LineStyle','-');

    hold off;
end % I