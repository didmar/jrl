package com.github.didmar.jrl.utils.plot;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import com.github.didmar.jrl.evaluation.valuefunction.VFunction;
import com.github.didmar.jrl.utils.plot.Gnuplot;

/**
 * Performs a contour plot of a Q-function.
 * @author Didier Marin
 */
public final class VFunctionPlot {
	
	/** Title of the plot */
	private final String title;
	/** the Q-function to plot */
	private final VFunction vFunction;
	/** State-space samples */
	private final double[][] xs;
	///** Action-space bounds */
	//private final double[][] uBounds;
	/** State-space dimension */
	private final int xDim;
	/** Used to store sample state values */
	private final double[] samples;
	/** The gnuplot console */ 
	private final Gnuplot console;
	/** Temporary file filename */
	private final String tmpFilename;
	
	public VFunctionPlot(String title, VFunction vFunction,
			double[][] xs) throws IOException {
		this.title = title;
		this.vFunction = vFunction;
		// assert xs.length > 0 
		this.xs = xs;
		//this.uBounds = uBounds; 
		xDim = this.xs[0].length;
		if(xDim != 1) {
			System.err.println("Only 1-dim state space is supported by"
					+java.lang.String.class.getName());
			System.exit(0);
		}
		
		console = new Gnuplot();
		console.execute("set xlabel 'state'");
		console.execute("set ylabel 'value'");
		
		File tmpfile;
		tmpfile = File.createTempFile(this.getClass().getSimpleName(),
				"gnuplot");
		tmpfile.deleteOnExit();
		tmpFilename = tmpfile.getAbsolutePath();
		
		samples = new double[xs.length];
	}
	
	/**
	 * Use it for discrete state spaces 
	 */
	public final void plotHistogram() {
		final double stepXDiv2 = (xs[1][0]-xs[0][0])/2;
		// Compute the action for each state sample
		for(int i = 0; i < xs.length; i++) {
			samples[i] = vFunction.get(xs[i]);
		}
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<xs.length; x++) {
				ps.print(Double.toString(xs[x][0]-stepXDiv2)+' '+Double.toString(samples[x]));
				ps.println();
				ps.print(Double.toString(xs[x][0]+stepXDiv2)+' '+Double.toString(samples[x]));
				ps.println();
			}
			ps.flush();
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		StringBuilder cmd = new StringBuilder(256);
		cmd.append("plot '");
		cmd.append(tmpFilename);
		cmd.append("' title '"+title+"' with lines");
		console.execute(cmd.toString());
	}
	
	public final void plot() {
		// Compute the action for each state sample
		for(int i = 0; i < xs.length; i++) {
			samples[i] = vFunction.get(xs[i]);
		}
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<xs.length; x++) {
				for (int i=0; i<xDim; i++) {
						ps.print(xs[x][i]);
						ps.print(' ');
				}
				ps.print(samples[x]);
				ps.println();
			}
			ps.flush();
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		StringBuilder cmd = new StringBuilder(256);
		cmd.append("plot '");
		cmd.append(tmpFilename);
		cmd.append("' title '"+title+"' with lines");
		console.execute(cmd.toString());
	}
}
