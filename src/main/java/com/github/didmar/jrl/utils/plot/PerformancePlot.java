package com.github.didmar.jrl.utils.plot;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import com.github.didmar.jrl.agent.PolicyAgent;
import com.github.didmar.jrl.environment.Environment;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.policy.ParametricPolicy;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.Gnuplot;

/**
 * Plot the performance function with respect to policy parameters.
 * 
 * @author Didier Marin
 */
public final class PerformancePlot {
	
	/** Title of the plot */
	private final String title;
	/** The gnuplot console */ 
	private final Gnuplot console;
	/** Name of the temporary file */
	private final String tmpFilename;
	
	private final String tmpFilename2;
	/** Policy for which we want to plot the performance */
	private final ParametricPolicy pol;
	/** Number of policy parameters */
	private final int nbPolParams;
	/** Environment in which to evaluate the policy */
	private final Environment env;
	/** Discount factor for the performance evaluation */
	private final DiscountFactor gamma;
	
	private double currentJ = 0.;
	private double[] currentTheta = null;
	
	public PerformancePlot(String title, ParametricPolicy pol, Environment env,
			DiscountFactor gamma) throws IOException {
		this.title = title;
		this.pol = pol;
		nbPolParams = pol.getParamsSize();
		if(nbPolParams > 2) {
			throw new RuntimeException("Cannot plot for more than 2 parameters !");
		}
		this.env = env;
		this.gamma = gamma;
		console = new Gnuplot();
		console.execute("set title '"+title+"'");
		switch(nbPolParams) {
			case 1 :
				console.execute("set style data lines");
				console.execute("set nokey");
				console.execute("set xlabel 'theta'");
				console.execute("set ylabel 'J(theta)'");
				break;
			case 2 :	
				console.execute("set xlabel 'theta_1'");
				console.execute("set ylabel 'theta_2'");
				console.execute("set pm3d map");
				break;
		}
		File tmpfile;
		tmpfile = File.createTempFile(this.getClass().getSimpleName(),
				"gnuplot");
		tmpfile.deleteOnExit();
		tmpFilename = tmpfile.getAbsolutePath();
		tmpfile = File.createTempFile(this.getClass().getSimpleName(),
				"gnuplot");
		tmpfile.deleteOnExit();
		tmpFilename2 = tmpfile.getAbsolutePath();
	}
	
	/**
	 * Compute the performance for a grid of parameters (i.e., 2-dim parameters)
	 * and store it into a temporary file
	 * @param thetas    grid of parameters
	 */
	public final void computePerf(double[] theta1s, int nbEpiPerParams,
			int maxT, double[] currentTheta) {
		// Backup the policy parameters
		double[] origParams = pol.getParams().clone();
		// Evaluate each pol params from the given grid
		double[] params = new double[1];
		double[] J = new double[theta1s.length];
		PolicyAgent agent = new PolicyAgent(pol);
		Logger log = new Logger(env.getXDim(), env.getUDim());
		env.addListener(log);
		
		for(int i=0; i<theta1s.length; i++) {
			log.reset();
			params[0] = theta1s[i];
			pol.setParams(params);
			for(int k=0; k<nbEpiPerParams; k++) {
				env.interact(agent, nbEpiPerParams, maxT);
			}
			J[i] = ArrUtils.mean(log.discountedReward(gamma));
			//System.out.print(J[i]+" ");
		}
		//System.out.println();
		
		if(currentTheta != null) {
			this.currentTheta = currentTheta;
			log.reset();
			params[0] = this.currentTheta[0];
			pol.setParams(params);
			for(int k=0; k<nbEpiPerParams; k++) {
				env.interact(agent, nbEpiPerParams, maxT);
			}
			currentJ  = ArrUtils.mean(log.discountedReward(gamma));
		} else {
			this.currentTheta = null;
		}
		
		env.removeListener(log);
		pol.setParams(origParams); // Restore old params
		
		// Write the performance to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<theta1s.length; x++) {
				ps.println(theta1s[x]+" "+J[x]);
			}
			ps.flush();
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// Write the current theta to a temporary file
		if(currentTheta != null) {
			try {
				PrintStream ps = new PrintStream(tmpFilename2);
				ps.println(currentTheta[0]+" "+currentJ);
				ps.flush();
				ps.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Compute the performance for a grid of parameters (i.e., 2-dim parameters)
	 * and store it into a temporary file
	 * @param thetas    grid of parameters
	 */
	public final void computePerf(double[] theta1s, double[] theta2s,
			int nbEpiPerParams, int maxT) {
		// Backup the policy parameters
		double[] origParams = pol.getParams().clone();
		// Evaluate each pol params from the given grid
		double[] params = new double[2];
		double[][] J = new double[theta1s.length][theta2s.length];
		PolicyAgent agent = new PolicyAgent(pol);
		Logger log = new Logger(env.getXDim(), env.getUDim());
		env.addListener(log);
		for(int i=0; i<theta1s.length; i++) {
			for(int j=0; j<theta2s.length; j++) {
				log.reset();
				params[0] = theta1s[i];
				params[1] = theta2s[j];
				pol.setParams(params);
				for(int k=0; k<nbEpiPerParams; k++) {
					env.interact(agent, nbEpiPerParams, maxT);
				}
				J[i][j] = ArrUtils.mean(log.discountedReward(gamma));
			}
		}
		env.removeListener(log);
		pol.setParams(origParams); // Restore old params
		
		// Write the performance to a temporary file
		try {
			PrintStream ps = new PrintStream(tmpFilename);
			for (int x=0; x<theta1s.length; x++) {
				for (int y=0; y<theta2s.length; y++) {
					ps.println(theta1s[x]+" "+theta2s[y]+" "+J[x][y]);
				}
				ps.println();
			}
			ps.flush();
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public final void plot() {
		StringBuilder cmd = new StringBuilder(1024);
		switch(nbPolParams) {
			case 1 :
				if(currentTheta != null) {
					// Also plot theta as a dot
					// TODO find the associated J
					//console.execute("plot '" + tmpFilename + "' using lines,"
					//	+ " '-' using 1:2 with points");
					//console.execute(currentTheta [0] + " " + currentJ);
					//console.execute("e");
					
					cmd.append("plot '" + tmpFilename + "' with lines, '-'"
							+ " using 1:2 with points\n"
							+ currentTheta[0] + " " + currentJ + "\n"
							+ "e\n");
					
//					cmd.append("plot '" + tmpFilename + "' with lines, '"
//							+ tmpFilename2 + "' using 1:2 with points");
					
					console.execute(cmd.toString());
				} else {
					cmd.append("plot '" + tmpFilename + "' with lines");
					console.execute(cmd.toString());
				}
				
				
				break;
			case 2 :
				cmd.append("splot '");
				cmd.append(tmpFilename);
				cmd.append("' title '"+title+"'");
				console.execute(cmd.toString());
				break;
		}
	}
	
}
