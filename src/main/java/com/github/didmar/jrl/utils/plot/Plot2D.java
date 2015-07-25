package com.github.didmar.jrl.utils.plot;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import org.eclipse.jdt.annotation.NonNullByDefault;
import org.eclipse.jdt.annotation.Nullable;

import com.github.didmar.jrl.utils.array.ArrUtils;

//TODO add the possibility to change plot style
//     (such as continuous line instead of dots)
/**
 * 2D plotting utility based on a Gnuplot console.
 * 
 * @author Didier Marin
 */
@NonNullByDefault
public final class Plot2D {
	
	/** Title of the plot */
	private final String title;
	/** The Gnuplot console */ 
	private final Gnuplot console;
	/** Name of the temporary file */
	private final String tmpFilename;
	
	public Plot2D(String title, String xlabel, String ylabel)
			throws IOException {
		this.title = title;
		console = new Gnuplot();
		console.execute("set xlabel '"+xlabel+"'");
		console.execute("set ylabel '"+ylabel+"'");
		final File tmpfile = File.createTempFile(
				this.getClass().getSimpleName(), "gnuplot");
		tmpfile.deleteOnExit();
		@Nullable final String _tmpFilename = tmpfile.getAbsolutePath();
		if(_tmpFilename != null) tmpFilename = _tmpFilename;
		else throw new RuntimeException("getAbsolutePath() returns null");
	}
	
	public final void plot(final double[] ys) {
		assert ys != null;
		plot(ArrUtils.linspace(0, ys.length-1, 1), ys);
	}
	
	public final void plot(final double[] xs, final double[] ys) {
		assert xs != null;
		assert ys != null;
		
		if(xs.length != ys.length) {
			throw new IllegalArgumentException(
					"xs and ys must have the same length");
		}
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int i=0; i<xs.length; i++) {
				ps.println(xs[i]+" "+ys[i]);
			}
			ps.flush();
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		StringBuilder sb = new StringBuilder(256);
		sb.append("plot '");
		sb.append(tmpFilename);
		sb.append("' using 1:2 title '"+title+"'");
		@Nullable final String cmd = sb.toString();
		if(cmd != null)	console.execute(cmd);
		else throw new RuntimeException("Failed to build gnuplot command");
	}
	
	public final void plot(final double[] xs, final double[][] ys) {
		assert xs != null;
		assert ys != null;
		assert ys[0] != null;
		assert ArrUtils.hasShape(ys,xs.length,ys[0].length);
		
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<xs.length; x++) {
				for (int y=0; y<ys.length; y++) {
					ps.print(xs[x]);
					for (int k = 0; k < ys[y].length; k++) {
						ps.print(" "+ys[y][k]);
					}
					ps.println();
				}
			}
			ps.flush();
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		StringBuilder sb = new StringBuilder(1024);
		sb.append("plot '");
		sb.append(tmpFilename);
		sb.append("' using 1:2 title '"+title+"'");
		for(int k=1; k<ys[0].length; k++) {
			sb.append(", '");
			sb.append(tmpFilename);
			sb.append("' using 1:"+(2+k)+" title '"+title+"'");
		}
		@Nullable final String cmd = sb.toString();
		if(cmd != null)	console.execute(cmd);
		else throw new RuntimeException("Failed to build gnuplot command");
	}
}
