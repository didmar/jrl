package com.github.didmar.jrl.utils.plot;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import com.github.didmar.jrl.utils.plot.Gnuplot;

/**
 * General contour plot tool.
 * @author Didier Marin
 */
public class ContourPlot {
	
	/** Title of the plot */
	private String title;
	/** The gnuplot console */ 
	private final Gnuplot console;
	/** Name of the temporary file */
	private final String tmpFilename;
	
	public ContourPlot(String title, String xlabel, String ylabel)
			throws IOException {
		this.title = title;
		console = new Gnuplot();
		console.execute("set xlabel '"+xlabel+"'");
		console.execute("set ylabel '"+ylabel+"'");
		console.execute("set pm3d map");
		File tmpfile;
		tmpfile = File.createTempFile(this.getClass().getSimpleName(),
				"gnuplot");
		tmpfile.deleteOnExit();
		tmpFilename = tmpfile.getAbsolutePath();
	}
	
	public void plot(double[] xs, double[] ys, double[][] zs) {
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<xs.length; x++) {
				for (int y=0; y<ys.length; y++) {
					ps.println(xs[x]+" "+ys[y]+" "+zs[x][y]);
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
	public void plotHistogram(double[] xs, double[] ys, double[][] zs) {
		final double stepXDiv2 = (xs.length > 1 ? (xs[1]-xs[0])/2 : 0.5);
		final double stepYDiv2 = (ys.length > 1 ? (ys[1]-ys[0])/2 : 0.5);
		
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<xs.length-0; x++) {
				for (int y=0; y<ys.length; y++) {
					ps.print(Double.toString(xs[x]-stepXDiv2)+' '+Double.toString(ys[y]-stepYDiv2)+' '+Double.toString(zs[x][y]));
					ps.println();
					ps.print(Double.toString(xs[x]-stepXDiv2)+' '+Double.toString(ys[y]+stepYDiv2)+' '+Double.toString(zs[x][y]));
					ps.println();
				}
				ps.println();
				for (int y=0; y<ys.length; y++) {
					ps.print(Double.toString(xs[x]+stepXDiv2)+' '+Double.toString(ys[y]-stepYDiv2)+' '+Double.toString(zs[x][y]));
					ps.println();
					ps.print(Double.toString(xs[x]+stepXDiv2)+' '+Double.toString(ys[y]+stepYDiv2)+' '+Double.toString(zs[x][y]));
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
