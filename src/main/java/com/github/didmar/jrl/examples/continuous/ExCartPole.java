package com.github.didmar.jrl.examples.continuous;

import java.io.IOException;

import com.github.didmar.jrl.agent.CEPS;
import com.github.didmar.jrl.agent.LearningAgent;
import com.github.didmar.jrl.agent.PGPE;
import com.github.didmar.jrl.agent.REINFORCE;
import com.github.didmar.jrl.agent.ac.BasicAC;
import com.github.didmar.jrl.agent.ac.NAC;
import com.github.didmar.jrl.agent.ac.TDNAC;
import com.github.didmar.jrl.agent.ac.VAC;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.environment.cartpole.CartPole;
import com.github.didmar.jrl.environment.cartpole.CartPoleDisplay;
import com.github.didmar.jrl.environment.cartpole.CartPole.CartPoleRewardType;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.vflearner.td.TDZero;
import com.github.didmar.jrl.features.CompatibleFeatures;
import com.github.didmar.jrl.features.RBFFeatures;
import com.github.didmar.jrl.policy.LinearGaussianPolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.stepsize.DecreasingStepSize;
import com.github.didmar.jrl.utils.CEParametersDistribution;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.PGPEParametersDistribution;
import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.TrajectoriesPlot;
import com.github.didmar.jrl.utils.plot.environment.EpisodePlayerUI;

/**
 * Test of the CartPole environment
 * @author Didier Marin
 */
public class ExCartPole {

	public static void main(String[] args) throws Exception {

		System.out.println("Which algorithm do you want to test ?");
		int agentChoice = Utils.chooseOne(new String[]
				{"REINFORCE","CEPS","BasicAC","VAC","TDNAC","NAC"});
		System.out.println("Show state-space trajectories ?");
		final boolean showTrajs = Utils.chooseOne(new String[]
				{"No","Yes"})==0 ? false : true;
		System.out.println("Display environment ?");
		final boolean dispEnv = Utils.chooseOne(new String[]
				{"No","Yes"})==0 ? false : true;
		//---[ Create a cart pole environment ]------------------------
		final CartPoleRewardType rewardType = CartPoleRewardType.REWARD_IF_TARGET;
		//final CartPoleRewardType rewardType = CartPoleRewardType.EASY_REWARD;
		//final CartPoleRewardType rewardType = CartPoleRewardType.PUNISH_IF_NOT_TARGET;
		final boolean randomStartState = true;
		final CartPole env = new CartPole(rewardType,randomStartState);
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
		final int maxT = 200; // maximum episode duration
		final int nLearningStep = 200;
		final int nEpiPerLearningStep = 50;
		final int nEpiPerTestStep = 50;

		//---[ Create features for value approx. and policy ]-------
		final int nbStateFeat = 6;
		final double[][] stateCenters = ArrUtils.buildGrid(xMin,xMax,nbStateFeat);
		final double[] sigma = new double[]{0.1,0.5,0.3,0.25};
		final RBFFeatures stateFeat = new RBFFeatures(stateCenters, sigma, false);
		final RBFFeatures polStateFeat = new RBFFeatures(stateCenters, sigma, true);

		//---[ Choose a policy ]------------------------------------------------
		// - Linear Gaussian with constant noise
		double[] noiseStdDev = null;
		if(agentChoice == 1 || agentChoice == 2) {
			// no noise for direct exploration methods
			noiseStdDev = ArrUtils.zeros(uDim);
		} else {
			noiseStdDev = ArrUtils.constvec(uDim,0.1); // For the rest
		}
		LinearGaussianPolicy pol = new LinearGaussianPolicy(polStateFeat,
					noiseStdDev, uMin, uMax, true);
		//pol.setParamsBounds(-1e15, +1e15);
		// - Linear Gaussian with state dependent noise
		//SDNoiseStdDev = sqrt(0.1)*ones(uDim)
		//pol = LinearGaussianPolicyWithSDNoise(polStateFeat, SDNoiseStdDev, \
		//                                      uBounds, redrawNoiseForEachAction=True)
		//env.addListener(pol) # to renew the SD noise at the begining of each episode

		//---[ Create a Value function approximator ]---------------------------
		LinearVFunction vFunction = new LinearVFunction(stateFeat);
		@SuppressWarnings("unused")
		LinearQFunction aFunction = new LinearQFunction(
				new CompatibleFeatures(pol, xDim, uDim),xDim,uDim);

		//System.out.println("Debug : "
		//		+Utils.toString(aFunction.getFeatures().phi(new double[]{0.5,0.1})));

		//---[ Create a TD Learner (for BasicAC, iNAC, VAC, ...) ]--------------

		// Create a step-size for TD
		//final ConstantStepSize TDstepSize = new ConstantStepSize(0.0001);
		//final DecreasingStepSize TDstepSize = new DecreasingStepSize(0.01,1000,2./3.);

		//final DecreasingStepSize[] twoTimescaleStepsSizes =
		//	DecreasingStepSize.createTwoTimescaleStepSizes(0.01,1e5,0.001,1e5);

		final DecreasingStepSize[] twoTimescaleStepsSizes =
			DecreasingStepSize.createTwoTimescaleStepSizes(0.001,1e10,0.0001,1e10);

		final DecreasingStepSize TDstepSize = twoTimescaleStepsSizes[0];

		final TDZero td = new TDZero(vFunction, TDstepSize, gamma);
		//final TDLambda td = new TDLambda(vFunction, TDstepSize, gamma, 0.5);

		// +++ Advantage TD Bootstrap
//		//QFunctionWithBaseline qFunction = new QFunctionWithBaseline(aFunction,
//		//		vFunction, +1);
//		QFunction qFunction = aFunction;
//		AdvantageTDBootstrap advTDBoot = new AdvantageTDBootstrap(aFunction, td,
//				twoTimescaleStepsSizes[1]);
//		env.addListener(advTDBoot);

		// +++ ILSTDAV
//		final ILSTDAV ilstdav = new ILSTDAV(aFunction, vFunction, gamma, 0.,
//				maxT*nEpiPerLearningStep, 0.001);
//		env.addListener(ilstdav);

		//---[ Create a Learning Agent ]----------------------------------------

		LearningAgent agent = null;

		switch(agentChoice) {
			case 0 : { // +++ REINFORCE
				final ConstantStepSize stepSize = new ConstantStepSize(0.001);
				//final DecreasingStepSize stepSize = new DecreasingStepSize(0.01,1000);
				agent = new REINFORCE(pol, xDim, uDim, stepSize, nEpiPerLearningStep, gamma);
				}
				break;
			case 1 : { // +++ CEPS
				final int nPolEvalPerUpdate = nEpiPerLearningStep;
				final int nEpiPerPolEval = 1;
				final int nbSelectedPol = (int) Math.ceil(nPolEvalPerUpdate*50./100.);
				agent = new CEPS(pol, nPolEvalPerUpdate,
						nEpiPerPolEval,	nbSelectedPol, gamma,
						ArrUtils.constvec(pol.getParamsSize(),100.), 0., false, true);
				}
				break;
			case 2 : { // +++ BasicAC
				final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
				agent = new BasicAC(pol, td, actorStepSize);
				}
				break;
			case 3 : { // +++ VAC
				final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
				agent = new VAC(pol, td, actorStepSize, xDim, uDim);
				}
				break;
			case 4 : { // +++ TDNAC
				final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
				final DiscountFactor kappa = new DiscountFactor(1);
				agent = new TDNAC(pol, td, actorStepSize, kappa, xDim, uDim);
				}
				break;
			case 5 : { // +++ NAC
				final ConstantStepSize nacStepSize = new ConstantStepSize(1.0);
				final DiscountFactor lambda = new DiscountFactor(0.5);
				final DiscountFactor kappa = new DiscountFactor(1);
				agent = new NAC(pol, stateFeat, nacStepSize,
						gamma, lambda, kappa,
						maxT*nEpiPerLearningStep, 0.001, xDim, uDim );
				aFunction = ((NAC)agent).getAFunction();
				vFunction = ((NAC)agent).getVFunction();
				}
				break;
		}
		if(agent == null) throw new RuntimeException();

		env.addListener(agent); // to learn, he must be listening to his environment !

		//---[ Create a logger that will store the sample trajectories ]--------
		final Logger log = new Logger(xDim, uDim,
				Math.max(nEpiPerLearningStep, nEpiPerTestStep));
		env.addListener(log); // make it listen to the environment

		// Plot the mean action from the barycentric interpolation policy
		//pol.plot( stateSampleGrid )
		// Plot the initial state-value approximation
		//td.vFunction.plot( stateSampleGrid )

		// For plotting
//		PolicyPlot polPlot = null;
//		try {
//			polPlot = new PolicyPlot(pol, stateSampleGrid, uBounds);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

//		QFunctionPlot qPlot = null;
//		try {
//			qPlot = new QFunctionPlot("A", aFunction, stateSampleGrid,
//					actionSampleGrid);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

//		VFunctionPlot vPlot = null;
//		try {
//			vPlot = new VFunctionPlot("V", vFunction, stateSampleGrid);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

		TrajectoriesPlot learnTrajPlot = null;
		TrajectoriesPlot testTrajPlot = null;
		if(showTrajs) {
			learnTrajPlot = new TrajectoriesPlot(
					"Learning trajectories", xDim, uDim, 0,
					xMin, xMax, uMin, uMax, null, maxT);
			testTrajPlot = new TrajectoriesPlot(
					"Testing trajectory", xDim, uDim, 0,
					xMin, xMax, uMin, uMax, null, maxT);
		}

		//---[ Create a display to visualize the test episodes ]----------------
		CartPoleDisplay disp = null;
		if(dispEnv) {
			disp = new CartPoleDisplay();
			disp.openInJFrame(640, 480, "Cart pole test episodes");
		}

		//---[ Learning loop ]--------------------------------------------------
		System.out.println( "Learning with "+agent.toString() );
		for(int i=0; i<nLearningStep; i++) {

			// Learn during nEpiPerLearningStep episodes
		    log.reset(); // clear the testing history
		    env.interact(agent,nEpiPerLearningStep,maxT);
//		    qPlot.plot();
//		    vPlot.plot();

		    if(learnTrajPlot != null && i % 10 == 0) {
		    	learnTrajPlot.plot(log.getEpisodes());
		    }

		    // Testing
		    log.reset(); // clear the learning history
		    env.removeListener(agent); // won't learn during testing

		    // Prepare the policy for testing : make it deterministic
		    if(pol instanceof LinearGaussianPolicy) {
		        pol.setSigma(ArrUtils.zeros(uDim));
		    }
		    //if isinstance(pol,LinearGaussianPolicyWithSDNoise):
		    //    pol.setSDNoiseStdDev(0.*SDNoiseStdDev)

		    //env.useRandomStartState(false);

		    if(dispEnv) env.addListener(disp);

		    // Test on nEpiPerTestStep episodes
		    env.interact(agent,nEpiPerTestStep,maxT);

		    if(dispEnv) env.removeListener(disp);

		    // Printing some stats
		    System.out.print("nepi=" + (i+1)*nEpiPerLearningStep
		          + " J=" + ArrUtils.mean(log.discountedReward(gamma)));

		    // Print some agent-specific stats
		    if(agent instanceof CEPS) {
		    	CEPS ceps = (CEPS) agent;
		    	CEParametersDistribution paramsDist = ceps.getParamsDist();
		    	double minSigma = ArrUtils.min(paramsDist.getSigma());
		    	double maxSigma = ArrUtils.max(paramsDist.getSigma());
		    	double meanSigma = ArrUtils.mean(paramsDist.getSigma());
		    	System.out.println(" CEPS sigma min="+minSigma+" mean="+meanSigma+" max="+maxSigma);
		    } else if(agent instanceof PGPE) {
		    	PGPE pgpe = (PGPE) agent;
		    	PGPEParametersDistribution paramsDist = pgpe.getParamsDist();
		    	double minSigma = ArrUtils.min(paramsDist.getSigma());
		    	double maxSigma = ArrUtils.max(paramsDist.getSigma());
		    	double meanSigma = ArrUtils.mean(paramsDist.getSigma());
		    	System.out.println(" PGPE sigma min="+minSigma+" mean="+meanSigma+" max="+maxSigma);
		    } else {
		    	System.out.println();
		    }

		    // Plot the policy and the trajectories
		    //polPlot.plot();
		    if(testTrajPlot != null && i % 10 == 0) {
		    	testTrajPlot.plot(log.getEpisodes());
		    }

		    // Prepare the policy for learning : restore the stochasticity
		    if(pol instanceof LinearGaussianPolicy) {
		        pol.setSigma(noiseStdDev);
		    }
		    //if isinstance(pol,LinearGaussianPolicyWithSDNoise):
		    //    pol.setSDNoiseStdDev(SDNoiseStdDev)

		    //env.useRandomStartState(randomStartState);

		    env.addListener(agent);
		}

		// Save the last series of test episodes to binary files
		int cpt = 1;
		for(Episode e : log.getEpisodes()) {
			e.writeToBinaryFile("cartpole_episode_"+cpt+".dat");
			cpt++;
		}

		// Start a GUI to inspect the last series of test episodes
		new EpisodePlayerUI("Cart pole last test episodes",
				new CartPoleDisplay(), log.getEpisodes(), 100, gamma);

		System.out.println("Press a key to terminate...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
