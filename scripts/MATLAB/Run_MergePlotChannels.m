clear;
close all;
clc;

% Base directory and filename.
baseDir = '../../data/SLFusion/match_single_line_mb_tsukuba/lc_0176/0050';

figACRef = plot_3_channel( baseDir, 'ACRef', 'ACRef' );
figACTst = plot_3_channel( baseDir, 'ACTst', 'ACTst' );
