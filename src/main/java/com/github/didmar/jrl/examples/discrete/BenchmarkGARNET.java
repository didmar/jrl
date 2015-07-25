package com.github.didmar.jrl.examples.discrete;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import com.github.didmar.jrl.agent.Agent;
import com.github.didmar.jrl.agent.ac.BasicAC;
import com.github.didmar.jrl.agent.ac.KNAC;
import com.github.didmar.jrl.agent.ac.NAC;
import com.github.didmar.jrl.agent.ac.TDNAC;
import com.github.didmar.jrl.agent.ac.VAC;
import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.environment.discrete.DiscreteMDPEnvironment;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.ParametricVFunction;
import com.github.didmar.jrl.evaluation.vflearner.td.TDZero;
import com.github.didmar.jrl.features.TabularStateActionFeatures;
import com.github.didmar.jrl.features.TabularStateFeatures;
import com.github.didmar.jrl.mdp.GARNETMDP;
import com.github.didmar.jrl.mdp.dp.PolicyIteration;
import com.github.didmar.jrl.policy.BoltzmannPolicy;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.stepsize.DecreasingStepSize;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Benchmark of some Actor-Critic methods on the GARNET environment.
 * @author Didier Marin
 */
public class BenchmarkGARNET {

	public static void main(String[] args) throws Exception {

		//---[ Create a GARNET environment ]------------------------------------
		final int n = 10;
		final int m = 5;
		final int b = 3;
		// Fix the random generator seed to get to same environment every time
		RandUtils.setSeed(1);
		GARNETMDP mdp = new GARNETMDP(n, m, b);
		DiscreteMDPEnvironment env = new DiscreteMDPEnvironment(mdp);
		RandUtils.setSeed(System.currentTimeMillis());
		final int xDim = env.getXDim();
		final int uDim = env.getUDim();

		//---[ Set the run parameters ]-----------------------------------------
		final DiscountFactor gamma = new DiscountFactor(0.95);
		final int maxT = 100; // maximum episode duration
		// indexes of the test to perform
		final int[] settings = {8, 9, 10};
		final int nSettings = settings.length;
		final int nTrials = 10;
		final int nLearningStep = 30;
		// Number of episodes for each learning step :
		// [1 2 3 ... 9 10 20 30 ... 90 100 200 ... ]
		final int[] nEpiPerLearningStep = new int[nLearningStep];
		for(int i=0; i<nLearningStep; i++) {
			nEpiPerLearningStep[i] = ((int)Math.pow(10, i/10)) * ((i % 10)+1);
			//System.out.println("i="+i+" : "+nEpiPerLearningStep[i]);
		}
		// Number of episodes for each testing step :
		final int nEpiPerTestStep = 100;

		// Test performance
		final double[][] perf = new double[nTrials][nLearningStep];

		//---[ Compute optimal policy performance using Policy Iteration ]------
		final int piMaxIter = 1000;
		PolicyIteration pi = new PolicyIteration(mdp, gamma, piMaxIter);
		//final int[] piPol = pi.getPol();
		final double[] piV = pi.getV();
		final double piJ = mdp.expectedDiscountedReward(piV);
		System.out.println("Policy Iteration J="+piJ);

		//---[ Create GARNET specific features ]--------------------------------
		final TabularStateActionFeatures stateActionFeat =
			new TabularStateActionFeatures(mdp);
		final TabularStateFeatures stateFeat =
			new TabularStateFeatures(mdp);

		String plotCommand = "./plot_perf.py";

		//---[ Settings loop ]--------------------------------------------------
		for(int set=0; set<nSettings; set++) {

			Agent agent = null;
			String agentName = null;

			//---[ Trials loop ]----------------------------------------------------
			for(int trial=0; trial<nTrials; trial++) {

				System.out.println("Trial "+(trial+1)+"/"+nTrials);

				//---[ Create a policy ]--------------------------------------------
				// Generate the set of all legal actions
				final double[][] actions = new double[m][1];
				for(int i=0; i<m; i++) {
					actions[i][0] = i;
				}
				final double temp = 1.;
				// Create a Boltzmann policy
				BoltzmannPolicy pol = new BoltzmannPolicy(stateActionFeat, actions, temp);

				//---[ Create a Value function approximator ]-----------------------
				final LinearVFunction vFunction = new LinearVFunction(stateFeat);

				//---[ Create an AC Agent ]-----------------------------------------

				// Get the AC according to the settings index
				agent = null;
				agentName = null;
				switch(settings[set]) {
					case 0 : {
						// +++ BasicAC
						final DecreasingStepSize[] twoTimescaleStepsSizes =
							DecreasingStepSize.createTwoTimescaleStepSizes(0.001,1e5,
									0.0001,1e5);
						final DecreasingStepSize TDstepSize = twoTimescaleStepsSizes[0];
						final TDZero td = new TDZero(vFunction, TDstepSize, gamma);
						//final TDLambda td = new TDLambda(vFunction, gamma, 0.5, TDstepSize);
						final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
						agent = new BasicAC(pol, td, actorStepSize);
						agentName = "BasicAC";
						break;
					}
					case 1 : {
						// +++ VAC
						final DecreasingStepSize[] twoTimescaleStepsSizes =
							DecreasingStepSize.createTwoTimescaleStepSizes(0.001,1e5,
									0.0001,1e5);
						final DecreasingStepSize TDstepSize = twoTimescaleStepsSizes[0];
						final TDZero td = new TDZero(vFunction, TDstepSize, gamma);
						//final TDLambda td = new TDLambda(vFunction, gamma, 0.5, TDstepSize);
						final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
						agent = new VAC(pol, td, actorStepSize, xDim, uDim);
						agentName = "VAC";
						break;
					}

					case 2 : {
						// +++ TDNAC 0.001 1e5 0.0001,1e5
						final DecreasingStepSize[] twoTimescaleStepsSizes =
							DecreasingStepSize.createTwoTimescaleStepSizes(0.001,1e5,
									0.0001,1e5);
						final DecreasingStepSize TDstepSize = twoTimescaleStepsSizes[0];
						final TDZero td = new TDZero(vFunction, TDstepSize, gamma);
						//final TDLambda td = new TDLambda(vFunction, gamma, 0.5, TDstepSize);
						final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
						final DiscountFactor kappa = new DiscountFactor(1);
						agent = new TDNAC(pol, td, actorStepSize, kappa, xDim, uDim);
						agentName = "TDNAC_0_001_1e5_0_0001_1e5";
						break;
					}
					case 3 : {
						// +++ TDNAC 0.01 1e5 0.001,1e5
						final DecreasingStepSize[] twoTimescaleStepsSizes =
							DecreasingStepSize.createTwoTimescaleStepSizes(0.01,1e5,
									0.001,1e5);
						final DecreasingStepSize TDstepSize = twoTimescaleStepsSizes[0];
						final TDZero td = new TDZero(vFunction, TDstepSize, gamma);
						//final TDLambda td = new TDLambda(vFunction, gamma, 0.5, TDstepSize);
						final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
						final DiscountFactor kappa = new DiscountFactor(1);
						agent = new TDNAC(pol, td, actorStepSize, kappa, xDim, uDim);
						agentName = "TDNAC_0_01_1e5_0_001_1e5";
						break;
					}
					case 4 : {
						// +++ TDNAC 0.01 1e7 0.001,1e7
						final DecreasingStepSize[] twoTimescaleStepsSizes =
							DecreasingStepSize.createTwoTimescaleStepSizes(0.01,1e7,
									0.001,1e7);
						final DecreasingStepSize TDstepSize = twoTimescaleStepsSizes[0];
						final TDZero td = new TDZero(vFunction, TDstepSize, gamma);
						//final TDLambda td = new TDLambda(vFunction, gamma, 0.5, TDstepSize);
						final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
						final DiscountFactor kappa = new DiscountFactor(1);
						agent = new TDNAC(pol, td, actorStepSize, kappa, xDim, uDim);
						agentName = "TDNAC_0_01_1e7_0_001_1e7";
						break;
					}
					case 5 : {
						// +++ TDNAC 0.01 1e7 0.01,1e7
						final DecreasingStepSize[] twoTimescaleStepsSizes =
							DecreasingStepSize.createTwoTimescaleStepSizes(0.01,1e7,
									0.01,1e7);
						final DecreasingStepSize TDstepSize = twoTimescaleStepsSizes[0];
						final TDZero td = new TDZero(vFunction, TDstepSize, gamma);
						//final TDLambda td = new TDLambda(vFunction, gamma, 0.5, TDstepSize);
						final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
						final DiscountFactor kappa = new DiscountFactor(1);
						agent = new TDNAC(pol, td, actorStepSize, kappa, xDim, uDim);
						agentName = "TDNAC_0_01_1e7_0_01_1e7";
						break;
					}
					case 6 : {
						// +++ TDNAC 0.1 1e7 0.1,1e7
						final DecreasingStepSize[] twoTimescaleStepsSizes =
							DecreasingStepSize.createTwoTimescaleStepSizes(0.1,1e7,
									0.1,1e7);
						final DecreasingStepSize TDstepSize = twoTimescaleStepsSizes[0];
						final TDZero td = new TDZero(vFunction, TDstepSize, gamma);
						//final TDLambda td = new TDLambda(vFunction, gamma, 0.5, TDstepSize);
						final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
						final DiscountFactor kappa = new DiscountFactor(1);
						agent = new TDNAC(pol, td, actorStepSize, kappa, xDim, uDim);
						agentName = "TDNAC_0_1_1e7_0_1_1e7";
						break;
					}
					case 7 : {
						// +++ TDNAC 0.1 1e15 0.01,1e15
						final DecreasingStepSize[] twoTimescaleStepsSizes =
							DecreasingStepSize.createTwoTimescaleStepSizes(0.1,1e15,
									0.1,1e15);
						final DecreasingStepSize TDstepSize = twoTimescaleStepsSizes[0];
						final TDZero td = new TDZero(vFunction, TDstepSize, gamma);
						//final TDLambda td = new TDLambda(vFunction, gamma, 0.5, TDstepSize);
						final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
						final DiscountFactor kappa = new DiscountFactor(1);
						agent = new TDNAC(pol, td, actorStepSize, kappa, xDim, uDim);
						agentName = "TDNAC_0_1_1e15_0_1_1e15";
						break;
					}

					case 8 : {
						// +++ NAC 1.0 0 0.99 0.001
						final ConstantStepSize stepSize =
							new ConstantStepSize(1.0);
						agent = new NAC(pol, stateFeat, stepSize, gamma,
								new DiscountFactor(0), new DiscountFactor(0.99),
								maxT, 0.001, xDim, uDim);
						agentName = "NAC_1.0_0_0.99_0.001";
						break;
					}

					case 9 : {
						// +++ NAC 1.0 0.5 0.99 0.001
						final ConstantStepSize stepSize =
							new ConstantStepSize(1.0);
						agent = new NAC(pol, stateFeat, stepSize, gamma,
								new DiscountFactor(0.5), new DiscountFactor(0.99),
								maxT, 0.001, xDim, uDim);
						agentName = "NAC_1.0_0.5_0.99_0.001";
						break;
					}

					case 10 : {
						// +++ NAC 1.0 0. 0.999 0.001
						final ConstantStepSize stepSize =
							new ConstantStepSize(1.0);
						agent = new NAC(pol, stateFeat, stepSize, gamma,
								new DiscountFactor(0), new DiscountFactor(0.999),
								maxT, 0.001, xDim, uDim);
						agentName = "NAC_1.0_0.5_0.999_0.001";
						break;
					}

					case 11 : {
						// +++ KNAC
						final DecreasingStepSize actorStepSize =
							new DecreasingStepSize(0.1,1e5);
						final DiscountFactor lambda = new DiscountFactor(0);
						final double P_evo_init = 0.;
						final double eta = 1e-5;
						final double P_obs_step = 1.;
						final double k = 0.1;
						final double sigma_squared = 1e-2;
						agent = new KNAC(pol, vFunction, actorStepSize, gamma,
								lambda, P_evo_init, eta, P_obs_step, k,
								sigma_squared, xDim, uDim);
						agentName = "KNAC";
						break;
					}

				}
				assert(agent != null);
				assert(agentName != null);
				// Check that he can listen to his environment
				assert(!(agent instanceof EnvironmentListener));

				// To learn, he must be listening to his environment !
				env.addListener((EnvironmentListener)agent);

				//---[ Create a logger that will store the sample trajectories ]----
				final Logger log = new Logger(xDim,uDim);
				env.addListener(log); // make it listen to the environment

				//---[ Learning loop ]----------------------------------------------
				System.out.println( "Learning with "+agentName );
				for(int i=0; i<nLearningStep; i++) {

					// Learn during nEpiPerLearningStep episodes
				    log.reset(); // clear the testing history
				    env.interact(agent,nEpiPerLearningStep[i],maxT);

				    // Testing during nEpiPerTestStep episodes
				    log.reset(); // clear the learning history
				    env.removeListener((EnvironmentListener)agent); // won't learn during testing
				    // TODO make the Boltzmann greedy for testing
				    env.interact(agent,nEpiPerTestStep,maxT);
				    // Compute and store the performance
				    perf[trial][i] = ArrUtils.mean(log.discountedReward(gamma));
				    // Print some infos
				    System.out.println("iter="+(i+1)+" J="+perf[trial][i]);
				    env.addListener((EnvironmentListener)agent); // restore the agent listening
				}
				log.reset();
				env.removeAllListener();
			} // end of trials loop

			// Write the performances in a file
			File file = new File("garnet_perf_"+agentName);
	        FileWriter fstream = null;
			try {
				fstream = new FileWriter(file);
			} catch (IOException e) {
				e.printStackTrace();
			}
	        BufferedWriter out = new BufferedWriter(fstream);

	        try {
	        	out.write("# Performance on the GARNET environment\n");
	        	out.write("# n="+n+", m="+m+", b="+b+"\n");
	        	out.write("# gamma="+gamma+"\n");
	        	out.write("# maxT="+maxT+"\n");
	        	out.write("# nTrials="+nTrials+"\n");
	        	out.write("# nLearningStep="+nLearningStep+"\n");
	        	//out.write("# nEpiPerLearningStep="+nEpiPerLearningStep+"\n");
	        	out.write("# nEpiPerTestStep="+nEpiPerTestStep+"\n");
	        	int totNLearningEpis = 0;
	        	for(int i=0; i<nLearningStep; i++) {
	        		totNLearningEpis += nEpiPerLearningStep[i];
	        		out.write(Integer.toString(totNLearningEpis));
	        		//out.write(Integer.toString((j+1)*nEpiPerLearningStep));
	        		if(i < nLearningStep-1) {
        				out.write(" ");
        			}
	        	}
	        	out.write("\n");
	        	for(int i=0; i<nTrials; i++) {
	        		for(int j=0; j<nLearningStep; j++) {
	        			out.write(Double.toString(perf[i][j]));
	        			if(j < nLearningStep-1) {
	        				out.write(" ");
	        			}
	        		}
	        		if(i < nTrials-1) {
	        			out.write("\n");
	        		}
		        }
		        out.close();
		        fstream.close();
	        } catch (IOException e) {
				e.printStackTrace();
			}

	        plotCommand = plotCommand.concat(" "+agentName+" garnet_perf_"+agentName);

		} // end of settings loop

		System.out.println("To plot the results, use command :\n"+plotCommand);
	}
}
