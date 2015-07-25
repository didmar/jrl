package com.github.didmar.jrl.examples.discrete;

import java.io.IOException;

import com.github.didmar.jrl.agent.SARSAPolicyAgent;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.environment.discrete.DiscreteMDPEnvironment;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.TabularQFunction;
import com.github.didmar.jrl.features.TabularStateActionFeatures;
import com.github.didmar.jrl.mdp.GARNETMDP;
import com.github.didmar.jrl.mdp.dp.PolicyIteration;
import com.github.didmar.jrl.policy.BoltzmannPolicyOverQ;
import com.github.didmar.jrl.stepsize.DecreasingStepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.QFunctionPlot;

/**
 * @author Didier Marin
 */
public class ExGARNET {

	public static void main(String[] args) throws Exception {

		//---[ Create the GARNET environment ]----------------------------------
		final int n = 5;
		final int m = 5;
		final int b = 3;
		GARNETMDP mdp = new GARNETMDP(n,m,b);
		final DiscountFactor gamma = new DiscountFactor(0.95);
		DiscreteMDPEnvironment env = new DiscreteMDPEnvironment(mdp);
		final double[][] states = mdp.statesGrid();
		final double[][] actions = mdp.actionsGrid();

		//---[ Compute the optimal policy through PI ]--------------------------

		final int piMaxIter = 100000;
		PolicyIteration pi = new PolicyIteration(mdp, gamma, piMaxIter);
		final double[] piV = pi.getV();
		final double[][] piQ = pi.getQ();
		final double piJ = mdp.expectedDiscountedReward(piV);
		System.out.println("Policy Iteration : J="+piJ);
		QFunctionPlot piQPlot = new QFunctionPlot("Opt Q-function with PI",
				new TabularQFunction(piQ), states, actions);
		piQPlot.plotHistogram();

		//---[ Set the run parameters ]-----------------------------------------
		final int maxT = 1000; // maximum episode duration
		final int nLearningStep = 100;
		final int nEpiPerLearningStep = 100;
		final int nEpiPerTestStep = 100;

		//---[ Create features for value approx. and policy ]-------------------
		final TabularStateActionFeatures stateActionFeat =
				new TabularStateActionFeatures(mdp);

		//---[ Create a state-action value function approximator ]--------------
		LinearQFunction qFunction = new LinearQFunction(stateActionFeat,1,1);

		//---[ Create a policy over Q ]----------------------------------------
		double temp = 10.;
		BoltzmannPolicyOverQ pol = new BoltzmannPolicyOverQ(qFunction, actions,
				temp);

		//---[ Create a SARSA Learner ]-----------------------------------------
		final DecreasingStepSize stepSize = new DecreasingStepSize(0.1, 1000.);
		final DiscountFactor lambda = new DiscountFactor(0);
		SARSAPolicyAgent agent = new SARSAPolicyAgent(pol, gamma, lambda,
				stepSize);

		//---[ Create a logger that will store the sample trajectories ]--------
		final Logger log = new Logger(1, 1,
				Math.max(nEpiPerLearningStep, nEpiPerTestStep));
		env.addListener(log); // make it listen to the environment

		QFunctionPlot qPlot = new QFunctionPlot("Q-function",
					qFunction, states, actions);

		//---[ Learning loop ]--------------------------------------------------
		System.out.println( "Learning with "+agent.toString() );
		for(int i=0; i<nLearningStep; i++) {

			// Learn during nEpiPerLearningStep episodes
		    log.reset(); // clear the testing history
		    env.interact(agent,nEpiPerLearningStep,maxT);
		    qPlot.plotHistogram();
		    //System.out.println("probsTable :\n"+Utils.toString(pol.getProbaTable(states)));

		    // Testing
		    log.reset(); // clear the learning history
		    env.removeListener(agent); // won't learn during testing

		    // Prepare the policy for testing : make it deterministic
		    if(pol instanceof BoltzmannPolicyOverQ) {
		        ((BoltzmannPolicyOverQ)pol).setTemp(0.01);
		    }

		    // Test on nEpiPerTestStep episodes
		    env.interact(agent,nEpiPerTestStep,maxT);

		    // Plotting and printing some stats
		    //System.out.println( log );
		    System.out.println("nepi=" + (i+1)*nEpiPerLearningStep
		          + " J=" + ArrUtils.mean(log.discountedReward(gamma)));

		    // Prepare the policy for learning : restore the stochasticity
		    if(pol instanceof BoltzmannPolicyOverQ) {
		        ((BoltzmannPolicyOverQ)pol).setTemp(temp);
		    }

		    env.addListener(agent);
		}

		// Save the last series of test episodes to binary files
		int cpt = 1;
		for(Episode e : log.getEpisodes()) {
			e.writeToBinaryFile("garnet_episode_"+cpt+".dat");
			cpt++;
		}

		System.out.println("Press a key to terminate...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

}
