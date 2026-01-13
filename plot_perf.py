#!/usr/bin/env python3

# Plot the performances from a file with the following format :
#    |# first commented line
#    |# ...
#    |# last commented line
#    |total_nb_epi_aft_iter1 total_nb_epi_aft_iter2 ... total_nb_epi_aft_iterM
#    |perf_trial1_iter1      perf_trial1_iter2      ... perf_trial1_iterM
#    |perf_trial2_iter1      perf_trial2_iter2      ... perf_trial2_iterM
#    |...                    ...                        ...
#    |perf_trialN_iter1      perf_trialN_iter2      ... perf_trialN_iterM
#
# See for example jrl.examples.BenchmarkGARNET

# TODO show the std dev

import sys
import os

# Use non-interactive backend if no display available
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

from numpy import empty, mean, std
from matplotlib.pyplot import figure, subplot, xlabel, ylabel, errorbar, legend, show, savefig

nPerfFiles = (len(sys.argv)-1)//2
if len(sys.argv) < 3 or ((len(sys.argv)-1)%2) != 0:
    print("Usage : plot_perf.py perf_title1 perf_filename1 [perf_title2 perf_filename2 ...]")
else:
    xlogscale = True # TODO should be an argument of the script
    fig = figure()
    if xlogscale:
        subplot(111, xscale="log")
    xlabel("Episodes")
    ylabel("Performance")
    legends = []
    for k in range(nPerfFiles):
        #print("- ",sys.argv[1+k*2])
        legends.append(sys.argv[1+k*2])
        lines = open(sys.argv[2+k*2], 'r').readlines()
        # Remove commented lines
        while lines[0][0] == '#':
            del lines[0]
        # Then, the first line contains the total number of episodes
        # after each iteration
        strNEpis = lines.pop(0).split(" ")
        nIters = len(strNEpis)
        nEpis = empty(nIters)
        for i in range(nIters):
            nEpis[i] = float(strNEpis[i])
        # Now we can go through the perf of each trial and each iteration
        nTrials = len(lines)
        perf = empty((nTrials,nIters))
        for i in range(nTrials):
            sp = lines[i].split(" ")
            for j in range(nIters):
                perf[i,j] = float(sp[j])
        #plot(nEpis,mean(perf, axis=0))
        errorbar(nEpis,mean(perf, axis=0), yerr=std(perf, axis=0), label=sys.argv[1+k*2])
    #legend(legends,loc='best')
    legend(loc='best')
    # Save to file if running headless, otherwise show
    if 'DISPLAY' not in os.environ:
        savefig('performance_plot.png', dpi=150, bbox_inches='tight')
        print("Plot saved to performance_plot.png")
    else:
        show()
