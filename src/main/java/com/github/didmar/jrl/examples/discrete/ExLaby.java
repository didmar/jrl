package com.github.didmar.jrl.examples.discrete;

import java.io.IOException;

import com.github.didmar.jrl.agent.QLearningPolicyAgent;
import com.github.didmar.jrl.agent.SARSAPolicyAgent;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.environment.discrete.DiscreteMDPEnvironment;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.features.TabularStateActionFeatures;
import com.github.didmar.jrl.mdp.LabyMDP;
import com.github.didmar.jrl.policy.BoltzmannPolicyOverQ;
import com.github.didmar.jrl.policy.EpsGreedyPolicyOverQ;
import com.github.didmar.jrl.policy.QFunctionBasedPolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Testing pure Critic methods on a labyrinth problem.
 * 
 * @author Didier Marin
 */
public class ExLaby {
	public static void main(String[] args) throws Exception {
		
		int policyChoice = 0;
		if(args.length == 0) {
			System.out.println("Type of policy ?\n");
			policyChoice = Utils.chooseOne(
					new String[]{"Boltzmann policy","Eps-greedy policy"});
		} else {
			policyChoice = Integer.parseInt(args[0]);
		}

		//---[ Create a labyrinth environment ]--------------------------------
		int width = 5;
		int height = 5;
		final LabyMDP mdp = new LabyMDP(width,height);
		mdp.setReward(1, 2, +1.0);
		mdp.setReward(3, 2, +0.5);
		mdp.setObstacle(0, 1);
		mdp.setObstacle(1, 1);
		mdp.setObstacle(2, 1);
		mdp.setObstacle(2, 2);
		final DiscreteMDPEnvironment env = new DiscreteMDPEnvironment(mdp);
		final int xDim = env.getXDim();
		final int uDim = env.getUDim();
		final double[][] states = mdp.statesGrid();
		final double[][] actions = mdp.actionsGrid();
		
		//---[ Set the run parameters ]-----------------------------------------
		final DiscountFactor gamma = new DiscountFactor(0.95);
		final int maxT = 1000; // maximum episode duration
		final int nLearningStep = 20;
		final int nEpiPerLearningStep = 10;
		final int nEpiPerTestStep = 50;

		//---[ Create features for value approx. and policy ]-------------------

		final TabularStateActionFeatures stateActionFeat
			= new TabularStateActionFeatures(mdp);
		
		//---[ Create a state-action value function approximator ]--------------
		LinearQFunction qFunction = new LinearQFunction(stateActionFeat,xDim,uDim);

		//---[ Create a policy over Q ]----------------------------------------
		QFunctionBasedPolicy pol = null;
		// +++ Boltzmann policy
		double temp = 100.;
		if(policyChoice == 0) {
			pol = new BoltzmannPolicyOverQ(qFunction, actions, temp);
		}
		// +++ Epsilon-greedy policy		
		double eps = 0.3;
		if(policyChoice == 1) {
			pol = new EpsGreedyPolicyOverQ(qFunction, actions, eps);
		}

		if(pol == null) {
			System.err.println("Invalid policy choice !");
			System.exit(1);
		}
		System.out.println("Using "+pol.toString());

		//---[ Create a Learner ]-----------------------------------------
//		final StepSize stepSize = new ConstantStepSize(0.01);
//		double lambda = 0.;
//		SARSAPolicyAgent agent = new SARSAPolicyAgent(pol, gamma, lambda,
//				stepSize);

		final StepSize stepSize = new ConstantStepSize(0.01);
		final DiscountFactor lambda = new DiscountFactor(0.5);
		QLearningPolicyAgent agent = new QLearningPolicyAgent(pol, actions,
				gamma, lambda, stepSize);
		
		//---[ Create a logger that will store the sample trajectories ]--------
		final Logger log = new Logger(xDim, uDim,
				Math.max(nEpiPerLearningStep, nEpiPerTestStep));
		env.addListener(log); // make it listen to the environment

		//---[ Learning loop ]--------------------------------------------------
		System.out.println( "Learning with "+agent.toString() );
		for(int i=0; i<nLearningStep; i++) {

			// Learn during nEpiPerLearningStep episodes
			log.reset(); // clear the testing history
			env.interact(agent,nEpiPerLearningStep,maxT);

			// Testing
			log.reset(); // clear the learning history
			env.removeListener(agent); // won't learn during testing

			// Prepare the policy for testing : make it deterministic
			if(pol instanceof BoltzmannPolicyOverQ) {
				((BoltzmannPolicyOverQ)pol).setTemp(0.01);
			}
			if(pol instanceof EpsGreedyPolicyOverQ) {
				((EpsGreedyPolicyOverQ)pol).setEps(0.);
			}

			//env.useRandomStartState(false);

			// Test on nEpiPerTestStep episodes
			env.interact(agent,nEpiPerTestStep,maxT);

			// Plotting and printing some stats
			//System.out.println( log );
			System.out.println("nepi=" + (i+1)*nEpiPerLearningStep
					+ " J=" + ArrUtils.mean(log.discountedReward(gamma)));
			//System.out.println("theta="+Utils.vectorToString(pol.getTheta()));
			//		    polPlot.plot();
			
			// Prepare the policy for learning : restore the stochasticity 
			if(pol instanceof BoltzmannPolicyOverQ) {
				((BoltzmannPolicyOverQ)pol).setTemp(temp);
			}
			if(pol instanceof EpsGreedyPolicyOverQ) {
				((EpsGreedyPolicyOverQ)pol).setEps(eps);
			}

			//env.useRandomStartState(randomStartState);

			env.addListener(agent);
		}

		// Save the last series of test episodes to binary files  
//		int cpt = 1;
//		for(Episode e : log.getEpisodes()) {
//			e.writeToBinaryFile("twostate_episode_"+cpt+".dat");
//			cpt++;
//		}

		double[][] probas = null;
		if(pol instanceof BoltzmannPolicyOverQ) {
			probas = ((BoltzmannPolicyOverQ)pol).getProbaTable(states);
		}
		if(pol instanceof EpsGreedyPolicyOverQ) {
			probas = ((EpsGreedyPolicyOverQ)pol).getProbaTable(states);
		}
		mdp.printLaby();
		System.out.println();
		mdp.printPolicy(probas);
		
		System.out.println("Press a key to terminate...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
