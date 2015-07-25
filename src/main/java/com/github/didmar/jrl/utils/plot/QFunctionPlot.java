package com.github.didmar.jrl.utils.plot;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.utils.plot.Gnuplot;

/**
 * Performs a contour plot of a Q-function for a 1-dim state space and
 * 1-dim action space.
 * @author Didier Marin
 */
public final class QFunctionPlot {
	
	/** Title of the plot */
	private final String title;
	/** the Q-function to plot */
	private final QFunction qFunction;
	/** State-space samples */
	private final double[][] xs;
	/** Action-space samples */
	private final double[][] us;
	///** Action-space bounds */
	//private final double[][] uBounds;
	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;
	/** used to store sample values (Q-values) */
	private final double[][] samples;
	/** The gnuplot console */ 
	private final Gnuplot console;
	/** Temporary file filename */
	private final String tmpFilename;
	
	public QFunctionPlot(String title, QFunction qFunction,
			double[][] xs, double[][] us) throws IOException {
		this.title = title;
		this.qFunction = qFunction;
		// assert xs.length > 0 
		this.xs = xs;
		this.us = us;
		//this.uBounds = uBounds; 
		xDim = this.xs[0].length;
		if(xDim != 1) {
			throw new RuntimeException("Only 1-dim state space is supported by "
					+java.lang.String.class.getName());
		}
		uDim = this.us[0].length;
		if(uDim != 1) {
			throw new RuntimeException("Only 1-dim action space is supported by "
					+java.lang.String.class.getName());
		}
		
		console = new Gnuplot();
		console.execute("set xlabel 'state'");
		console.execute("set ylabel 'action'");
		console.execute("set title '"+title+"'");
		console.execute("set pm3d map");
		
		File tmpfile;
		tmpfile = File.createTempFile(this.getClass().getSimpleName(),
				"gnuplot");
		tmpfile.deleteOnExit();
		tmpFilename = tmpfile.getAbsolutePath();
		
		samples = new double[xs.length][us.length];
	}
	
	public final void plot() {
		// Compute the action for each state sample
		for(int i = 0; i < xs.length; i++) {
			for(int j = 0; j < us.length; j++) {
				samples[i][j] = qFunction.get(xs[i],us[j]);
			}
		}
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<xs.length; x++) {
				for (int u=0; u<us.length; u++) {
					for (int i=0; i<xDim; i++) {
						ps.print(xs[x][i]);
						ps.print(' ');
					}
					for (int j=0; j<uDim; j++) {
						ps.print(us[u][j]);
						ps.print(' ');
					}
					ps.print(samples[x][u]);
					ps.println();
				}
				ps.println();
			}
			ps.flush();
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		StringBuilder cmd = new StringBuilder(256);
		cmd.append("splot '");
		cmd.append(tmpFilename);
		cmd.append("' title '"+title+"'");
		console.execute(cmd.toString());
	}
	
	/**
	 * Use it for discrete state and action spaces 
	 */
	public final void plotHistogram() {
		final double stepXDiv2 = (xs.length > 1 ? (xs[1][0]-xs[0][0])/2 : 0.5);
		final double stepUDiv2 = (us.length > 1 ? (us[1][0]-us[0][0])/2 : 0.5);
		// Compute the action for each state sample
		for(int i = 0; i < xs.length; i++) {
			for(int j = 0; j < us.length; j++) {
				samples[i][j] = qFunction.get(xs[i],us[j]);
			}
		}
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<xs.length-0; x++) {
				for (int u=0; u<us.length; u++) {
					ps.print(Double.toString(xs[x][0]-stepXDiv2)+' '+Double.toString(us[u][0]-stepUDiv2)+' '+Double.toString(samples[x][u]));
					ps.println();
					ps.print(Double.toString(xs[x][0]-stepXDiv2)+' '+Double.toString(us[u][0]+stepUDiv2)+' '+Double.toString(samples[x][u]));
					ps.println();
				}
				ps.println();
				for (int u=0; u<us.length; u++) {
					ps.print(Double.toString(xs[x][0]+stepXDiv2)+' '+Double.toString(us[u][0]-stepUDiv2)+' '+Double.toString(samples[x][u]));
					ps.println();
					ps.print(Double.toString(xs[x][0]+stepXDiv2)+' '+Double.toString(us[u][0]+stepUDiv2)+' '+Double.toString(samples[x][u]));
					ps.println();
				}
				ps.println();
			}
			ps.flush();
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		StringBuilder cmd = new StringBuilder(256);
		cmd.append("splot '");
		cmd.append(tmpFilename);
		cmd.append("' title '"+title+"'");
		console.execute(cmd.toString());
	}
}
