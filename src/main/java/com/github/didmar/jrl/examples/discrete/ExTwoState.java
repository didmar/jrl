package com.github.didmar.jrl.examples.discrete;

import java.io.IOException;

import com.github.didmar.jrl.agent.QLearningPolicyAgent;
import com.github.didmar.jrl.agent.SARSAPolicyAgent;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.environment.discrete.DiscreteMDPEnvironment;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.features.TabularStateActionFeatures;
import com.github.didmar.jrl.mdp.TwoStateMDP;
import com.github.didmar.jrl.policy.BoltzmannPolicyOverQ;
import com.github.didmar.jrl.policy.EpsGreedyPolicyOverQ;
import com.github.didmar.jrl.policy.QFunctionBasedPolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Testing Q-Learning and SARSA on the TwoState.
 * 
 * @author Didier Marin
 */
public class ExTwoState {
	public static void main(String[] args) throws Exception {

		if(args.length == 0) {
			System.out.println("Usage : TestTwoStateSARSA policyChoice\n"
					+"policyChoice=0 => Boltzmann policy\n"
					+"policyChoice=1 => Eps-greedy policy");
			System.exit(0);
		}

		int policyChoice = Integer.parseInt(args[0]);

		//---[ Create a two state environment ]--------------------------------
		final TwoStateMDP mdp = new TwoStateMDP();
		final DiscreteMDPEnvironment env = new DiscreteMDPEnvironment(mdp);
		final int xDim = env.getXDim();
		final int uDim = env.getUDim();

		//---[ Set the run parameters ]-----------------------------------------
		final DiscountFactor gamma = new DiscountFactor(0.95);
		final int maxT = 1000; // maximum episode duration
		final int nLearningStep = 100;
		final int nEpiPerLearningStep = 1;
		final int nEpiPerTestStep = 1;

		//---[ Create features for value approx. and policy ]-------------------

		final TabularStateActionFeatures stateActionFeat
			= new TabularStateActionFeatures(mdp);

		//---[ Create a state-action value function approximator ]--------------
		LinearQFunction qFunction = new LinearQFunction(stateActionFeat,xDim,uDim);

		//---[ Create a policy over Q ]----------------------------------------
		double[][] actions = {{0}, {1.}};
		QFunctionBasedPolicy pol = null;
		// +++ Botltzmann policy
		double temp = 1.;
		if(policyChoice == 0) {
			pol = new BoltzmannPolicyOverQ(qFunction, actions, temp);
		}
		// +++ Epsilon-greedy policy		
		double eps = 0.2;
		if(policyChoice == 1) {
			pol = new EpsGreedyPolicyOverQ(qFunction, actions, eps);
		}

		if(pol == null) {
			System.err.println("Invalid policy choice !");
			System.exit(1);
		}
		System.out.println("Using "+pol.toString());

		//---[ Create a SARSA Learner ]-----------------------------------------
//		final StepSize stepSize = new ConstantStepSize(0.01);
//		double lambda = 0.9;
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

		System.out.println("Press a key to terminate...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
