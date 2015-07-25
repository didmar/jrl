package com.github.didmar.jrl.utils.plot;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;

import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.plot.Gnuplot;

/**
 * Plot the samples of some episodes.
 * @author Didier Marin
 */
public final class TrajectoriesPlot {

	/** Indicates what to draw :
	 * 0 is x[0] given time,
	 * 1 is x[1] given x[0],
	 * 2 is u[0] given time,
	 * 3 is u[1] given u[0],
	 * 4 is u[0] given x[0],
	 * 5 is r    given time. */
	private final int mode;
	/** The gnuplot console */
	private final Gnuplot console;
	/** Name of the temporary file */
	private final String tmpFilename;

	public TrajectoriesPlot(String title, int xDim, int uDim, int mode,
			double[] xMin, double[] xMax, double[] uMin, double[] uMax,
			double[] rBounds, int maxT) throws IOException {
		this.mode = mode;
		console = new Gnuplot();
		console.execute("set style data lines");
		console.execute("set nokey");
		console.execute("set title '"+title+"'");

		switch(mode) {
			case 0 :
				console.execute("set xlabel 'time'");
				console.execute("set ylabel 'state'");
				if(maxT > 0) {
					console.execute("set xrange [0:"+maxT+"]");
				}
				if(xMin != null && xMax != null) {
					console.execute("set yrange ["+xMin[0]+":"+xMax[0]+"]");
				}
				break;
			case 1 :
				if(xDim < 2) {
					throw new IllegalArgumentException("State-space must have at least 2 dimensions");
				}
				console.execute("set xlabel 'state(1)'");
				console.execute("set ylabel 'state(2)'");
				if(xMin != null && xMax != null) {
					console.execute("set xrange ["+xMin[0]+":"+xMax[0]+"]");
					console.execute("set yrange ["+xMin[1]+":"+xMax[1]+"]");
				}
				break;
			case 2 :
				console.execute("set xlabel 'time'");
				console.execute("set ylabel 'action'");
				if(maxT > 0) {
					console.execute("set xrange [0:"+maxT+"]");
				}
				if(uMin != null && uMax != null) {
					console.execute("set yrange ["+uMin[0]+":"+uMax[0]+"]");
				}
				break;
			case 3 :
				if(uDim < 2) {
					throw new IllegalArgumentException("Action-space must have at least 2 dimensions");
				}
				console.execute("set xlabel 'action(1)'");
				console.execute("set ylabel 'action(2)'");
				if(uMin != null && uMax != null) {
					console.execute("set xrange ["+uMin[0]+":"+uMax[0]+"]");
					console.execute("set yrange ["+uMin[1]+":"+uMax[1]+"]");
				}
				break;
			case 4 :
				console.execute("set xlabel 'state'");
				console.execute("set ylabel 'action'");
				if(xMin != null && xMax != null) {
					console.execute("set xrange ["+xMin[0]+":"+xMax[0]+"]");
				}
				if(uMin != null && uMax != null) {
					console.execute("set yrange ["+uMin[0]+":"+uMax[0]+"]");
				}
				break;
			case 5 :
				console.execute("set xlabel 'time'");
				console.execute("set ylabel 'reward'");
				if(maxT > 0) {
					console.execute("set xrange [0:"+maxT+"]");
				}
				if(rBounds != null) {
					console.execute("set yrange ["+rBounds[0]+":"+rBounds[1]+"]");
				}
				break;
			default :
				throw new IllegalArgumentException("Unknow mode");
		}

		File tmpfile;
		tmpfile = File.createTempFile(this.getClass().getSimpleName(),
				"gnuplot");
		tmpfile.deleteOnExit();
		tmpFilename = tmpfile.getAbsolutePath();
	}

	public final void plot(List<Episode> epis) {
		int maxT = epis.get(0).getT();
		for(Episode e : epis) {
			if(e.getT() > maxT) {
				maxT = e.getT();
			}
		}
		// Write the samples to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for(int t=0; t<maxT; t++) {
				StringBuilder line = new StringBuilder(1024);
				for(int i=0; i<epis.size(); i++)
				{
					double[] x, u = null;
					double r = 0.;
					int time = t;
					if(t <= epis.get(i).getT()-1) {
						x = epis.get(i).getX()[t];
						u = epis.get(i).getU()[t];
						r = epis.get(i).getR()[t];
					} else {
						x = epis.get(i).getX()[epis.get(i).getT()-1];
						u = epis.get(i).getU()[epis.get(i).getT()-1];
						r = epis.get(i).getR()[epis.get(i).getT()-1];
						time = epis.get(i).getT()-1;
					}
					switch(mode) {
						case 0 :
							line.append(time+" "+x[0]); break;
						case 1 :
							line.append(x[0]+" "+x[1]); break;
						case 2 :
							line.append(time+" "+u[0]); break;
						case 3 :
							line.append(u[0]+" "+u[1]); break;
						case 4 :
							line.append(x[0]+" "+u[0]); break;
						case 5 :
							line.append(time+" "+r); break;
					}
					if(i<epis.size()-1) {
						line.append(" ");
					}
				}
				line.append("\n");
				ps.print(line.toString());
			}
			ps.flush();
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

		StringBuilder cmd = new StringBuilder(1024);
		cmd.append("plot ");
		for(int i=0; i<epis.size()-1; i++) {
			cmd.append("'"+tmpFilename+"'"
					+ " using "+(i*2+1)+":"+(i*2+2)+", ");
		}
		cmd.append("'"+tmpFilename+"'"
				+ " using "+((epis.size()-1)*2+1)+":"+((epis.size()-1)*2+2));
		console.execute(cmd.toString());
	}
}
