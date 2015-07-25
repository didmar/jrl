package com.github.didmar.jrl.examples.continuous;

import java.io.IOException;

import com.github.didmar.jrl.agent.QLearningPolicyAgent;
import com.github.didmar.jrl.agent.SARSAPolicyAgent;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.environment.acrobot.Acrobot;
import com.github.didmar.jrl.environment.acrobot.AcrobotDisplay;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.features.TileGridFeatures;
import com.github.didmar.jrl.policy.BoltzmannPolicyOverQ;
import com.github.didmar.jrl.policy.EpsGreedyPolicyOverQ;
import com.github.didmar.jrl.policy.QFunctionBasedPolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.stepsize.DecreasingStepSize;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.environment.EpisodePlayer;
import com.github.didmar.jrl.utils.plot.environment.EpisodePlayerUI;

/**
 * Testing the {@link Acrobot} environment.
 * @author Didier Marin
 */
public class ExAcrobot {

	public static void main(String[] args) throws Exception {

		int policyChoice = 0;
		if(args.length != 1) {
			System.out.println("Which policy to use ?");
			int agentChoice = Utils.chooseOne(new String[]
					{"Boltzmann policy","Eps-greedy policy"});
		} else {
			policyChoice = Integer.parseInt(args[0]);
		}

		//---[ Create an acrobot environment ]----------------------------------
		final double difficulty = 0.25; // For Sutton's book setting, use 0.75
		final Acrobot env = new Acrobot(difficulty);
		final int xDim = env.getXDim();
		final int uDim = env.getUDim();
		final double[] xMin = env.getXMin();
		final double[] xMax = env.getXMax();
		final double[] uMin = env.getUMin();
		final double[] uMax = env.getUMax();

		//final double[][] stateSampleGrid = Utils.buildGrid(xMin,xMax,11);
		//final double[][] actionSampleGrid = Utils.buildGrid(uMin,uMax,11);

		//---[ Set the run parameters ]-----------------------------------------
		final DiscountFactor gamma = new DiscountFactor(1);
		final int maxT = 1000; // maximum episode duration
		final int nLearningStep = 1000;
		final int nEpiPerLearningStep = 1;
		final int nEpiPerTestStep = 0;

		//---[ Create features for value approx. and policy ]-------------------
		//final int nbStateFeat = 6;
		//final double[][] stateCenters = Utils.buildGrid(xMin,xMax,nbStateFeat);
		//final double[] sigma = new double[]{0.1,0.1,0.1,0.1};
		//final RBFFeatures stateFeat = new RBFFeatures(stateCenters, sigma, false);
		//final RBFFeatures polStateFeat = new RBFFeatures(stateCenters, sigma, true);

		final int[] nbStateActionFeat = {6,6,7,7,3};
		final double[] xuMin = new double[xDim+uDim];
		final double[] xuMax = new double[xDim+uDim];
		System.arraycopy(xMin, 0, xuMin, 0, xDim);
		System.arraycopy(uMin, 0, xuMin, xDim, uDim);
		System.arraycopy(xMax, 0, xuMax, 0, xDim);
		System.arraycopy(uMax, 0, xuMax, xDim, uDim);
		final TileGridFeatures stateActionFeat = new TileGridFeatures(
				xuMin, xuMax, nbStateActionFeat);

		//---[ Create a state-action value function approximator ]--------------
		LinearQFunction qFunction = new LinearQFunction(
				stateActionFeat, xDim, uDim);

		//---[ Create a policy over Q ]-----------------------------------------
		int nbActions = 3;
		double[][] actions = ArrUtils.buildGrid(uMin, uMax, nbActions);
		QFunctionBasedPolicy pol = null;
		// +++ Botltzmann policy
		double temp = 1.;
		if(policyChoice == 0) {
			pol = new BoltzmannPolicyOverQ(qFunction, actions, temp);
		}
		// +++ Epsilon-greedy policy
		double eps = 0.; // 0 will work since Q initialization is optimistic
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
		final DiscountFactor lambda = new DiscountFactor(0.9);
		QLearningPolicyAgent agent = new QLearningPolicyAgent(pol, actions,
				gamma, lambda, stepSize);

		//---[ Create a logger that will store the sample trajectories ]--------
		final Logger log = new Logger(xDim, uDim,
				Math.max(nEpiPerLearningStep, nEpiPerTestStep));
		env.addListener(log); // make it listen to the environment

		//---[ Create a display to visualize the test episodes ]----------------
		AcrobotDisplay disp = new AcrobotDisplay(env);
	    disp.openInJFrame(640, 480, "Acrobot test episodes");
	    env.addListener(disp);

		//---[ Learning loop ]--------------------------------------------------
		System.out.println( "Learning with "+agent.toString() );
		for(int i=0; i<nLearningStep; i++) {

			// Learn during nEpiPerLearningStep episodes
		    log.reset(); // clear the testing history
		    env.interact(agent,nEpiPerLearningStep,maxT);
//		    qPlot.plot();
//		    vPlot.plot();

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
		int cpt = 1;
		for(Episode e : log.getEpisodes()) {
			e.writeToBinaryFile("acrobot_episode_"+cpt+".dat");
			cpt++;
		}

		// Start a GUI to inspect the last series of test episodes
		//new EpisodePlayerUI(new AcrobotDisplay(),
		//		log.getEpisodes(), 100);

		System.out.println("Press a key to terminate...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
