clear;
close all;
clc;

% WORK_DIR = '../../data/SLFusion/match_single_line_gradient_cost';
% WORK_DIR = '../../data/SLFusion/match_single_line_05_cost';
WORK_DIR = '../../data/SLFusion/match_single_line_mb_tsukuba';

TRUE_DISP_FN = 'TrueDisp.pgm';
ROW_IDX      = 145;

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
plot( costs(:, 1), '-*' );
xlabel('x location');
ylabel('disparity of minimum cost');
title('Disparity of minimum cost along single line');

% Load the true disparity.
td = imread([WORK_DIR, '/', TRUE_DISP_FN]);
hold on
plot( td(ROW_IDX, :) / 8, 'o-r' );
hold off

