package com.github.didmar.jrl.utils.plot;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import com.github.didmar.jrl.policy.Policy;
import com.github.didmar.jrl.utils.plot.Gnuplot;

public class PolicyPlot {
	
	/** The policy to plot **/
	private final Policy pol;
	/** State-space samples */
	private final double[][] xs;
	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;
	/** Used to store sample values (actions) */
	private final double[][] samples;
	/** The gnuplot console */ 
	private final Gnuplot console;
	/** Temporary file filename */
	private final String tmpFilename;
	
	public PolicyPlot(Policy pol, double[][] xs, double[] uMin, double[] uMax)
			throws IOException {
		if(xs.length <= 0) {
			throw new IllegalArgumentException("At least one state sample is required");
		}
		this.pol = pol;
		this.xs = xs;
		xDim = this.xs[0].length;
		if(xDim != 1) {
			throw new IllegalArgumentException("Only 1-dim state-space is supported");
		}
		uDim = uMin.length;
		if(uDim != 1) {
			throw new IllegalArgumentException("Only 1-dim action-space is supported");
		}
		
		console = new Gnuplot();
		console.execute("set grid");
		console.execute("set yrange["+ uMin[0] +":"+ uMax[0] +"]");
		console.execute("set xlabel 'state'");
		console.execute("set ylabel 'action'");
		console.execute("set style data lines");
		
		File tmpfile;
		tmpfile = File.createTempFile(this.getClass().getSimpleName(),
				"gnuplot");
		tmpfile.deleteOnExit();
		tmpFilename = tmpfile.getAbsolutePath();
		
		samples = new double[xs.length][uDim];
	}
	
	public void plot() {
		// Compute the action for each state sample
		for(int i = 0; i < xs.length; i++) {
			pol.computePolicyDistribution(xs[i]);
			double[] u = pol.drawAction();
			for(int j=0; j<uDim; j++) {
				samples[i][j] = u[j];
			}
		}
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<xs.length; x++) {
				for (int i=0; i<xDim; i++) {
					ps.print(xs[x][i]);
					ps.print(' ');
				}
				for (int y = 0; y < uDim; y++) {
					if (y > 0) {
						ps.print(' ');
					}
					ps.print(samples[x][y]);
				}
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
		cmd.append("' title 'mean action' using 1:2");
		console.execute(cmd.toString());
	}
}
