package com.github.didmar.jrl.examples.discrete;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import com.github.didmar.jrl.agent.CEPS;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.environment.discrete.DiscreteMDPEnvironment;
import com.github.didmar.jrl.mdp.TwoStateMDP;
import com.github.didmar.jrl.mdp.dp.PolicyEvaluation;
import com.github.didmar.jrl.policy.BagnellTwoStatePolicy;
import com.github.didmar.jrl.policy.ComplexTwoStatePolicy;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.ContourPlot;
import com.github.didmar.jrl.utils.plot.Gnuplot;

/**
 * Testing Policy Search methods on the TwoState.
 *
 * @author Didier Marin
 */
public class ExTwoStatePS {
	public static void main(String[] args) throws Exception {

		//---[ Create a point mass environment ]--------------------------------
		final TwoStateMDP mdp = new TwoStateMDP();
		final DiscreteMDPEnvironment env = new DiscreteMDPEnvironment(mdp);
		final int xDim = env.getXDim();
		final int uDim = env.getUDim();

		//---[ Set the run parameters ]-----------------------------------------
		final DiscountFactor gamma = new DiscountFactor(0.95);
		final int maxT = 100; // maximum episode duration
		final int nLearningStep = 100000;
		final int nEpiPerLearningStep = 100;

		//---[ Create a policy ]------------------------------------------------

		//BagnellTwoStatePolicy pol = new BagnellTwoStatePolicy(1.);
		ComplexTwoStatePolicy pol = new ComplexTwoStatePolicy();

		//---[ Sample the performance function and plot it ]--------------------

		final double[][] states
			= ArrUtils.buildGrid(ArrUtils.zeros(1), ArrUtils.ones(1), 2);
		double[] theta0s = ArrUtils.linspace(-5., +5., 101);
		double[] theta1s = ArrUtils.linspace(-5., +5., 101);
		double[][] J = new double[theta0s.length][theta1s.length];
		for(int i=0; i<theta0s.length; i++) {
			for(int j=0; j<theta1s.length; j++) {
				pol.setParams(new double[]{theta0s[i],theta1s[j]});
				double[][] probTable = pol.getProbaTable(states);
				PolicyEvaluation pe = new PolicyEvaluation(mdp, probTable, gamma);
				J[i][j] = mdp.expectedDiscountedReward(pe.getV());
			}
		}

		ContourPlot cp = new ContourPlot("Performance J","theta_0","theta_1");
		cp.plot(theta0s,theta1s,J);

		//---[ Set the initial policy parameters ]------------------------------

		//pol.setParams(new double[]{4.,4.});
		//pol.setParams(new double[]{0.,4.});
		//pol.setParams(new double[]{0.,0.});
		pol.setParams(new double[]{4.,0.});

		//---[ Create a learning agent ]----------------------------------------

		// +++ CEPS
		int nPolEvalPerUpdate = 20;
		int nEpiPerPolEval = 5;
		int nSelectedPol = 10;
		double[] sigma = ArrUtils.constvec(2, 1.);
		double noise = 0.0001;
		// Variant 1 : starts from the current policy parameters
		CEPS agent = new CEPS(pol, nPolEvalPerUpdate, nEpiPerPolEval,
				nSelectedPol, gamma, sigma, noise, false, false);
		// Variant 2 : starts from a uniform distribution over policy parameters
//		CEPS agent = new CEPS(pol, xDim, uDim, nPolEvalPerUpdate, nEpiPerPolEval,
//				nSelectedPol, gamma, new double[][]{{-5,+5},{-5,+5}}, noise, false, false);

		env.addListener(agent);

		//---[ Create a logger that will store the sample trajectories ]--------
		final Logger log = new Logger(xDim,uDim);
		env.addListener(log); // make it listen to the environment

		//---[ Plotting stuff ]-------------------------------------------------

		CEPSPlot polPlot = (new ExTwoStatePS()).new CEPSPlot(agent);

		//---[ Learning loop ]--------------------------------------------------
		System.out.println( "Learning with "+agent.toString() );
		for(int i=0; i<nLearningStep; i++) {
			// Learning
			log.reset(); // clear the history
			env.interact(agent,nEpiPerLearningStep,maxT);
			polPlot.plot();
			// Testing
			log.reset(); // clear the history
			env.removeListener(agent);
			env.interact(agent,100,maxT);
			System.out.println("nepi=" + (i+1)*nEpiPerLearningStep
			          + " J=" + ArrUtils.mean(log.discountedReward(gamma)));
			env.addListener(agent);
			Thread.sleep(500);
		}

		System.out.println("Press a key to terminate...");
		try {
			cp.plot(theta0s,theta1s,J);
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Plot the CEPS policy params distribution and its samples.
	 * @author Didier Marin
	 */
	public class CEPSPlot {

		private final CEPS ceps;
		/** The gnuplot console */
		private final Gnuplot console;
		/** Name of the temporary file */
		private final String tmpFilename;
		private final String tmpFilename2;

		public CEPSPlot(CEPS ceps) throws IOException {
			this.ceps = ceps;
			console = new Gnuplot();
			console.execute("set title 'CEPS'");
			console.execute("set xlabel 'theta_0'");
			console.execute("set ylabel 'theta_1'");
			console.execute("set xrange [-5:+5]");
			console.execute("set yrange [-5:+5]");
			//console.execute("set pointsize 5");
			File tmpfile = File.createTempFile(this.getClass().getSimpleName(),
					"gnuplot");
			tmpfile.deleteOnExit();
			tmpFilename = tmpfile.getAbsolutePath();
			File tmpfile2 = File.createTempFile(this.getClass().getSimpleName(),
					"gnuplot");
			tmpfile2.deleteOnExit();
			tmpFilename2 = tmpfile2.getAbsolutePath();
		}

		public void plot() {

			// Plot the policy parameters distribution
			double[] mean = ceps.getParamsDist().getMean();
			double[] sigma = ceps.getParamsDist().getSigma();
			try {
				PrintStream ps = new PrintStream(tmpFilename);
				int nbPts = 100;
				double step = 2*Math.PI / ((double)nbPts-1);
				for (int i=0; i<nbPts; i++) {
					double x = mean[0] + sigma[0] * Math.cos(step*i);
					double y = mean[1] + sigma[1] * Math.sin(step*i);
					ps.println(x+" "+y);
				}
				ps.flush();
				ps.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

			// Plot the samples
			double[][] thetas = ceps.getThetas();
			try {
				PrintStream ps = new PrintStream(tmpFilename2);
				for (int i=0; i<thetas.length; i++) {
					ps.println(thetas[i][0]+" "+thetas[i][1]);
				}
				ps.flush();
				ps.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

			StringBuilder cmd = new StringBuilder(1024);
			cmd.append("plot '");
			cmd.append(tmpFilename);
			cmd.append("' using 1:2 title '' with lines lt 2, ");
			cmd.append("'");
			cmd.append(tmpFilename2);
			cmd.append("' using 1:2 title '' with points lt 3");
			console.execute(cmd.toString());
		}
	}
}
