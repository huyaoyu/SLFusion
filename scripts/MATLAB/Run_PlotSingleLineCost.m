clear;
close all;
clc;

% WORK_DIR = '../../data/SLFusion/match_single_line_gradient_cost';
% WORK_DIR = '../../data/SLFusion/match_single_line_05_cost';
WORK_DIR = '../../data/SLFusion/match_single_line_mb_tsukuba';

TRUE_DISP_FN = 'TrueDisp.pgm';
ROW_IDX      = 145;

FONT_SIZE_LABEL = 8;
FONT_SIZE_TITLE = 8;
FONT_SIZE_TICK  = 8;

% List all the file with extension .dat.

files = dir( [WORK_DIR, '/*.dat'] );

nFiles = length(files);

% Pre-allocate.
costs = zeros(nFiles, 2);

% Loop over the files.
fprintf('Processing %d files...\n', nFiles);

for I = 1:1:nFiles
    fs = files(I);
    fn = [ WORK_DIR, '/', fs.name];
    
    fprintf('%s\n', fn);
    
    % Load the file.
    c = load([ WORK_DIR, '/', fs.name]);
    
    % Find the minimum cost.
    [ minC, idxMinC ] = min(c(:, 2));
    
    % Retrieve the disparity value of the minimum cost.
    d = c(idxMinC, 1);
    
    costs(I, :) = [ d, minC ];
end % I

% Plot the minimun cost line.
h = figure('Name', 'Disparity', 'NumberTitle', 'off');
plot( costs(:, 1), '-*' );
xlabel('x location', 'FontSize', FONT_SIZE_LABEL);
ylabel('disparity of minimum cost', 'FontSize', FONT_SIZE_LABEL);
title('Disparity of minimum cost along single line', 'FontSize', FONT_SIZE_TITLE);
set(gca, 'FontSize', FONT_SIZE_TICK);

ylim([0, 45]);

% Load the true disparity.
td = imread([WORK_DIR, '/', TRUE_DISP_FN]);
hold on
plot( td(ROW_IDX, :) / 8, 'o-r' );
hold off

legendString = {
    'BWM';
    'True'
};

legend(legendString);

% Save the figure as an image.
h.PaperUnits = 'centimeters';
h.PaperPosition = [0 0 8 6];
print('DispairtyAlongOneLine', '-dtiff', '-r300');
print('DispairtyAlongOneLine', '-dpng', '-r300');
