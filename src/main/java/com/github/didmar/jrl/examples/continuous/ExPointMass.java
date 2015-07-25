package com.github.didmar.jrl.examples.continuous;

import java.io.IOException;

import com.github.didmar.jrl.agent.CEPS;
import com.github.didmar.jrl.agent.LearningAgent;
import com.github.didmar.jrl.agent.PGPE;
import com.github.didmar.jrl.agent.PolicyAgent;
import com.github.didmar.jrl.agent.REINFORCE;
import com.github.didmar.jrl.agent.SARSAPolicyAgent;
import com.github.didmar.jrl.agent.ac.BasicAC;
import com.github.didmar.jrl.agent.ac.NAC;
import com.github.didmar.jrl.agent.ac.TDNAC;
import com.github.didmar.jrl.agent.ac.VAC;
import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.environment.PointMass;
import com.github.didmar.jrl.environment.PointMass.PointMassRewardType;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.evaluation.vflearner.lstd.ILSTDAV;
import com.github.didmar.jrl.evaluation.vflearner.td.AdvantageTDBootstrap;
import com.github.didmar.jrl.evaluation.vflearner.td.TDLambda;
import com.github.didmar.jrl.evaluation.vflearner.td.TDZero;
import com.github.didmar.jrl.features.CompatibleFeatures;
import com.github.didmar.jrl.features.RBFFeatures;
import com.github.didmar.jrl.features.TileGridFeatures;
import com.github.didmar.jrl.policy.EpsGreedyPolicyOverQ;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.policy.LinearGaussianPolicy;
import com.github.didmar.jrl.policy.ParametricPolicy;
import com.github.didmar.jrl.policy.Policy;
import com.github.didmar.jrl.policy.QFunctionBasedPolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.stepsize.DecreasingStepSize;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.CEParametersDistribution;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.PGPEParametersDistribution;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.PolicyPlot;
import com.github.didmar.jrl.utils.plot.QFunctionPlot;
import com.github.didmar.jrl.utils.plot.TrajectoriesPlot;
import com.github.didmar.jrl.utils.plot.VFunctionPlot;

/**
 * Learn an optimal policy on the Point Mass problem
 * @author Didier Marin
 */
@SuppressWarnings("unused")
public class ExPointMass {

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		//---[ Create a point mass environment ]--------------------------------
		final double[] x0 = ArrUtils.constvec(1, 0.25);
		final double[] xtarget = ArrUtils.constvec(1, 0.75);
		final PointMassRewardType rewardType = PointMassRewardType.GOAL_REWARD; //PointMassRewardType.COST;
		final boolean randomStartState = false;
		final PointMass env = new PointMass(x0, xtarget, rewardType,
				randomStartState);
		final int xDim = env.getXDim();
		final int uDim = env.getUDim();
		final double[] xMin = env.getXMin();
		final double[] xMax = env.getXMax();
		final double[] uMin = env.getUMin();
		final double[] uMax = env.getUMax();

		final double[][] stateSampleGrid = ArrUtils.buildGrid(xMin,xMax,101);
		final double[][] actionSampleGrid = ArrUtils.buildGrid(uMin,uMax,101);

		//---[ Set the run parameters ]-----------------------------------------
		final DiscountFactor gamma = new DiscountFactor(0.95);
		final int maxT = 100; // maximum episode duration
		final int nLearningStep = 10000;
		final int nEpiPerLearningStep = 1000;
		final int nEpiPerTestStep = 100;

		//---[ Create features for value approx. and policy ]-------------------
		final int nbStateFeat = 11;
		final double[][] stateCenters = ArrUtils.buildGrid(xMin,xMax,nbStateFeat);
		final double[] sigma = ArrUtils.constvec(xDim,0.01);
		final RBFFeatures stateFeat = new RBFFeatures(stateCenters, sigma, false);
		final RBFFeatures polStateFeat = new RBFFeatures(stateCenters, sigma, true);

		// For epsilon-greedy policy and SARSA
		final int[] nbStateActionFeat = {21,11};
		final double[] xuMin = new double[xDim+uDim];
		final double[] xuMax = new double[xDim+uDim];
		System.arraycopy(xMin, 0, xuMin, 0, xDim);
		System.arraycopy(uMin, 0, xuMin, xDim, uDim);
		System.arraycopy(xMax, 0, xuMax, 0, xDim);
		System.arraycopy(uMax, 0, xuMax, xDim, uDim);
		final TileGridFeatures stateActionFeat
			= new TileGridFeatures(xuMin, xuMax, nbStateActionFeat);

		//---[ Choose a policy ]------------------------------------------------
		// - Linear Gaussian with constant noise
		//   - For parameter-based methods : FD, iFD, CEPS, PGPE
		final double[] noiseStdDev = ArrUtils.zeros(uDim);
		//   - For the action-based methods : REINFORCE, Actor-Critics...
//		final double[] noiseStdDev = Utils.constvec(uDim,0.1);
		Policy pol = new LinearGaussianPolicy(polStateFeat, noiseStdDev, uMin, uMax, true);
		// - Epsilon-greedy policy over Q : SARSA
//		double[][] actions = {{-0.5}, {-0.4}, {-0.3}, {-0.2}, {-0.1}, {0.},
//				{+0.1}, {+0.2}, {+0.3}, {+0.4}, {+0.5}};
		double eps = 0.1;
//		LinearQFunction qFunction = new LinearQFunction(stateActionFeat,xDim,uDim);
//		QFunctionBasedPolicy pol = new EpsGreedyPolicyOverQ(qFunction, actions, eps);

		//---[ Create a Value function approximator ]---------------------------
		LinearVFunction vFunction = new LinearVFunction(stateFeat);
//		LinearQFunction aFunction = new LinearQFunction(
//				new CompatibleFeatures(pol, xDim, uDim),xDim,uDim);

		//---[ Create a TD Learner (for BasicAC, iNAC, VAC, ...) ]--------------

		// Create a step-size for TD
		//final ConstantStepSize TDstepSize = new ConstantStepSize(0.0001);
		//final DecreasingStepSize TDstepSize = new DecreasingStepSize(0.01,1000,2./3.);

		final DecreasingStepSize[] twoTimescaleStepsSizes =
			DecreasingStepSize.createTwoTimescaleStepSizes(0.01,1e5,0.001,1e5);

//		final DecreasingStepSize[] twoTimescaleStepsSizes =
//			DecreasingStepSize.createTwoTimescaleStepSizes(0.001,1e10,0.0001,1e10);

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

		//---[ Create an AC Agent ]---------------------------------------------

		// +++ Dummy
//		final PolicyAgent agent = new PolicyAgent(pol);

		// +++ REINFORCE
//		final ConstantStepSize stepSize = new ConstantStepSize(0.001);
//		//final DecreasingStepSize stepSize = new DecreasingStepSize(0.01,1000);
//		final LearningAgent agent = new REINFORCE(pol, xDim, uDim, stepSize, nEpiPerLearningStep, gamma);

		// +++ PGPE
//		final LearningAgent agent = new PGPE((ParametricPolicy)pol, gamma, nEpiPerLearningStep,
//				Utils.constvec(((ParametricPolicy)pol).getParamsSize(),0.01), 0.01,
//				new ConstantStepSize(0.001));

		// +++ CEPS
		final int nPolEvalPerUpdate = nEpiPerLearningStep;
		final int nEpiPerPolEval = 1;
		assert(nEpiPerPolEval * nPolEvalPerUpdate == nEpiPerLearningStep);
		final int nbSelectedPol = (int) Math.ceil(nPolEvalPerUpdate*10./100.);
		final LearningAgent agent = new CEPS((ParametricPolicy)pol,
				nPolEvalPerUpdate, nEpiPerPolEval,	nbSelectedPol, gamma,
				ArrUtils.constvec(((ParametricPolicy)pol).getParamsSize(),0.1),
				1e-2, false, false);

		// +++ CMAESPS
//		final int nPolEvalPerUpdate = nEpiPerLearningStep;
//		final int nEpiPerPolEval = 1;
//		assert(nEpiPerPolEval * nPolEvalPerUpdate == nEpiPerLearningStep);
//		final int nbSelectedPol = (int) Math.ceil(nPolEvalPerUpdate*10./100.);
//		final LearningAgent agent = new CMAESPS((ParametricPolicy)pol,
//				xDim, uDim, nPolEvalPerUpdate, nEpiPerPolEval, gamma,
//				Utils.constvec(((ParametricPolicy)pol).getParamsSize(),0.1));

		// +++ BasicAC
//		final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
//		final LearningAgent agent = new BasicAC((ILogDifferentiablePolicy)pol, td, actorStepSize);

		// +++ VAC
//		final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
//		final LearningAgent agent = new VAC((ILogDifferentiablePolicy)pol,
//				td, actorStepSize, xDim, uDim);

		// +++ TDNAC
//		final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
//		double kappa = 0.999;
//		final LearningAgent agent = new TDNAC((ILogDifferentiablePolicy)pol,
//				td, actorStepSize, kappa, xDim, uDim);

		// +++ NAC
//		final ConstantStepSize nacStepSize = new ConstantStepSize(1.0);
//		double lambda = 0.5, kappa = 1., diagAinv0 = 100.;
//		final LearningAgent agent = new NAC(pol, stateFeat, nacStepSize, gamma,
//				lambda, kappa, maxT*nEpiPerLearningStep, diagAinv0, xDim, uDim );
//		aFunction = agent.getAFunction();
//		vFunction = agent.getVFunction();

		// +++ SARSA
//		final StepSize stepSize = new ConstantStepSize(0.01);
//		double lambda = 0.;
//		LearningAgent agent = new SARSAPolicyAgent(pol, gamma, lambda,
//				stepSize);

		env.addListener(agent); // To learn, he must be listening to his environment !

		//---[ Create a logger that will store the sample trajectories ]--------
		final Logger log = new Logger(xDim,uDim);
		env.addListener(log); // make it listen to the environment

		// Plot the mean action from the barycentric interpolation policy
		//pol.plot( stateSampleGrid )
		// Plot the initial state-value approximation
		//td.vFunction.plot( stateSampleGrid )

		// For plotting
		PolicyPlot polPlot = null;
		try {
			polPlot = new PolicyPlot(pol, stateSampleGrid, uMin, uMax);
		} catch (IOException e) {
			e.printStackTrace();
		}

//		QFunctionPlot qPlot = null;
//		try {
//			qPlot = new QFunctionPlot("A", aFunction, stateSampleGrid,
//					actionSampleGrid);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//
//		VFunctionPlot vPlot = null;
//		try {
//			vPlot = new VFunctionPlot("V", vFunction, stateSampleGrid);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}

		TrajectoriesPlot trajPlot = new TrajectoriesPlot("Testing trajectories",
				xDim, uDim, 0, xMin, xMax, uMin, uMax, null, maxT);

		//---[ Learning loop ]--------------------------------------------------
		System.out.println( "Learning with "+agent.toString() );
		for(int i=0; i<nLearningStep; i++) {

			// Learn during nEpiPerLearningStep episodes
		    log.reset(); // clear the testing history
		    env.interact(agent,nEpiPerLearningStep,maxT);

		    // Plotting
//		    qPlot.plot();
//		    vPlot.plot();

		    // Testing
		    log.reset(); // clear the learning history
		    env.removeListener(agent); // won't learn during testing

		    // Prepare the policy for testing : make it deterministic
		    if(pol instanceof LinearGaussianPolicy) {
		        ((LinearGaussianPolicy)pol).setSigma(ArrUtils.zeros(uDim));
		    }
		    if(pol instanceof EpsGreedyPolicyOverQ) {
		        ((EpsGreedyPolicyOverQ)pol).setEps(0.);
		    }

		    // Test on nEpiPerTestStep episodes
		    env.interact(agent,nEpiPerTestStep,maxT);

		    // Print some stats
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

		    // Plot the policy
		    polPlot.plot();

		    trajPlot.plot(log.getEpisodes());

		    // Prepare the policy for learning : restore the stochasticity
		    if(pol instanceof LinearGaussianPolicy) {
		        ((LinearGaussianPolicy)pol).setSigma(noiseStdDev);
		    }
		    if(pol instanceof EpsGreedyPolicyOverQ) {
		        ((EpsGreedyPolicyOverQ)pol).setEps(eps);
		    }

		    //if isinstance(pol,LinearGaussianPolicyWithSDNoise):
		    //    pol.setSDNoiseStdDev(SDNoiseStdDev)
		    env.addListener(agent);
		}
	}
}
