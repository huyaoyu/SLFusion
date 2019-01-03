clear;
close all;
clc;

% Base directory and filename.
baseDir = '../../data/SLFusion/match_single_line_mb_tsukuba/lc_0134/0032';
titleCell = { 'winRef_0134'; 'winTst_0134_d0032'; 'ACRef_0134'; 'ACTst_0134_d0032' };
printFn   = { 'FigWinRef_0134'; 'FigWinTst_0134_d0032'; 'FigACRef_0134'; 'FigACTst_0134_d0032' };

FLAG_CIELAB = 1;

% Plot two figures.
figWinRef = plot_3_channel( baseDir, 'winRef', 'winRef', FLAG_CIELAB );
figWinTst = plot_3_channel( baseDir, 'winTst', 'winTst', FLAG_CIELAB );
figACRef  = plot_3_channel( baseDir, 'ACRef', 'ACRef', FLAG_CIELAB );
figACTst  = plot_3_channel( baseDir, 'ACTst', 'ACTst', FLAG_CIELAB );

% Format the figures.
figCell   = { figWinRef; figWinTst; figACRef; figACTst };

FONT_SIZE_TITLE = 8;
FONT_SIZE_LABEL = 8;
FONT_SIZE_TICK  = 8;

nFigs = size( figCell, 1 );

for I = 1:1:nFigs
    add_rectangle_3d(figCell{I, 1}, 19, -19, 2, 1, -1, [1, 0, 0]);
    
    set( figCell{I, 1}.Children.XLabel, 'FontSize', FONT_SIZE_LABEL );
    set( figCell{I, 1}.Children.YLabel, 'FontSize', FONT_SIZE_LABEL );
    set( figCell{I, 1}.Children, 'FontSize', FONT_SIZE_TICK );
    
    set( figCell{I, 1}.Children.Title, 'String', titleCell{I, 1} );
    set( figCell{I, 1}.Children.Title, 'FontSize', FONT_SIZE_TITLE );
    set( figCell{I, 1}.Children.Title, 'Interpreter', 'none' );
    
    % Print these two image.
    figCell{I, 1}.PaperUnits = 'centimeters';
    figCell{I, 1}.PaperPosition = [0 0 8 6];
    print( figCell{I, 1}, printFn{I, 1}, '-dpng',  '-r300' );
    print( figCell{I, 1}, printFn{I, 1}, '-dtiff', '-r300' );
end % I
