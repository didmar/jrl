package com.github.didmar.jrl.examples.continuous;

import java.io.IOException;

import com.github.didmar.jrl.agent.Agent;
import com.github.didmar.jrl.agent.CEPS;
import com.github.didmar.jrl.agent.PGPE;
import com.github.didmar.jrl.agent.REINFORCE;
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
import com.github.didmar.jrl.evaluation.vflearner.td.TD;
import com.github.didmar.jrl.evaluation.vflearner.td.TDZero;
import com.github.didmar.jrl.features.CompatibleFeatures;
import com.github.didmar.jrl.features.RBFFeatures;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.policy.LinearGaussianPolicy;
import com.github.didmar.jrl.policy.ParametricPolicy;
import com.github.didmar.jrl.policy.SharedParamsLGPolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.stepsize.DecreasingStepSize;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.PerformancePlot;
import com.github.didmar.jrl.utils.plot.PolicyPlot;
import com.github.didmar.jrl.utils.plot.QFunctionPlot;
import com.github.didmar.jrl.utils.plot.VFunctionPlot;

/**
 * Point Mass problem using a randomly generated policy,
 * where the number of parameters can be chosen.
 *
 * @author Didier Marin
 */
public class ExPMRandGenPol {

	public static void main(String[] args) throws Exception {

		System.out.println("Which algorithm do you want to test ?");
		int agentChoice = Utils.chooseOne(new String[]
				{"PGPE","CEPS","REINFORCE","BasicAC","VAC","TDNAC","NAC"});
		System.out.println("Display policy ?");
		final boolean dispPolicy = Utils.chooseOne(new String[]
				{"No","Yes"})==0 ? false : true;

		//---[ Create a point mass environment ]--------------------------------
		final double[] x0 = ArrUtils.constvec(1,0.25);
		final double[] xtarget = ArrUtils.constvec(1,0.75);
		final PointMassRewardType rewardType = PointMassRewardType.COST;
		final boolean randomStartState = false;
		final PointMass env = new PointMass(x0, xtarget, rewardType,
				randomStartState);
		final int xDim = env.getXDim();
		final int uDim = env.getUDim();
		final double[] xMin = env.getXMin();
		final double[] xMax = env.getXMax();
		final double[][] xBounds = new double[xDim][2];
		for(int i=0; i<xDim; i++) {
			xBounds[i][0] = xMin[i];
			xBounds[i][1] = xMax[i];
		}
		final double[] uMin = new double[]{-1.};
		final double[] uMax = new double[]{+1.};
		final double[][] uBounds = new double[uDim][2];
		for(int i=0; i<uDim; i++) {
			uBounds[i][0] = uMin[i];
			uBounds[i][1] = uMax[i];
		}

		final double[][] stateSampleGrid = ArrUtils.buildGrid(xMin,xMax,101);
		final double[][] actionSampleGrid = ArrUtils.buildGrid(uMin,uMax,101);

		//---[ Set the run parameters ]-----------------------------------------
		final DiscountFactor gamma = new DiscountFactor(0.95);
		final int maxT = 100; // maximum episode duration
		final int nLearningStep = 1000;
		final int nEpiPerLearningStep = 100;
		final int nEpiPerTestStep = 100;

		//---[ Generate features for policy ]-----------------------------------
		final int nbPolParams = 2;
		final int nbStateFeatPerParam = 11;
		final int nbStateFeatTotal = nbPolParams * nbStateFeatPerParam;
		// Randomly placed RBFs
		final double[][] stateCenters = new double[nbStateFeatTotal][xDim];
		for (int i = 0; i < stateCenters.length; i++) {
			for (int j = 0; j < xDim; j++) {
				stateCenters[i][j] = xMin[j] + (xMax[j]-xMin[j])*Math.random();
			}
		}
		final double[] sigma = ArrUtils.constvec(xDim,0.01);
		final RBFFeatures polStateFeat = new RBFFeatures(stateCenters, sigma, true);
		final RBFFeatures stateFeat = polStateFeat;

		//---[ Create a policy ]------------------------------------------------
		double[] noiseStdDev = null;
		// - Linear Gaussian with constant noise
		if(agentChoice <= 1) {
			//   - For parameter-perturbating methods (CEPS, PGPE, FD, iFD)
			noiseStdDev = ArrUtils.zeros(uDim);
		} else {
			//   - For the action-perturbating methods (REINFORCE, Actor-Critics)
			noiseStdDev = ArrUtils.constvec(uDim,0.1);
		}
		SharedParamsLGPolicy pol = new SharedParamsLGPolicy(
				polStateFeat, noiseStdDev, uMin, uMax, nbPolParams);

		//---[ Create a Value function approximator ]---------------------------
		LinearVFunction vFunction = new LinearVFunction(stateFeat);
		LinearQFunction aFunction = new LinearQFunction(
				new CompatibleFeatures((ILogDifferentiablePolicy)pol, xDim, uDim),
				xDim,uDim);

		//---[ Create a TD Learner (for BasicAC, iNAC, VAC, ...) ]--------------

		// Create a step-size for actor-critic methods
		StepSize TDstepSize = null;
		StepSize actorStepSize = null;
		TD td = null;
		if(agentChoice > 2) {
			TDstepSize = new ConstantStepSize(0.0001);
			//TDstepSize = new DecreasingStepSize(0.01,1000,2./3.);

			final DecreasingStepSize[] twoTimescaleStepsSizes =
				DecreasingStepSize.createTwoTimescaleStepSizes(
						0.01,1e5,0.001,1e5);
						//0.001,1e10,0.0001,1e10);
			TDstepSize = twoTimescaleStepsSizes[0];
			actorStepSize = twoTimescaleStepsSizes[1];

			td = new TDZero(vFunction, TDstepSize, gamma);
			//td = new TDLambda(vFunction, TDstepSize, gamma, 0.5);

			// +++ Advantage TD Bootstrap
			//QFunctionWithBaseline qFunction = new QFunctionWithBaseline(aFunction,
			//		vFunction, +1);
			QFunction qFunction = aFunction;
			AdvantageTDBootstrap advTDBoot = new AdvantageTDBootstrap(
					aFunction, td, twoTimescaleStepsSizes[1]);

			// +++ ILSTDAV
			final ILSTDAV ilstdav = new ILSTDAV(aFunction, vFunction, gamma,
					new DiscountFactor(0), maxT*nEpiPerLearningStep, 0.001);
		}

		//---[ Plot the performance function ]----------------------------------
		PerformancePlot perfPlot = new PerformancePlot(
				"Performance", pol, env, gamma);
		double[] theta1s = ArrUtils.linspace(0., 1., 101);
		double[] theta2s = ArrUtils.linspace(0., 1., 101);
		perfPlot.computePerf(theta1s, theta2s, 1, maxT);
		perfPlot.plot();

		//---[ Create an Agent ]------------------------------------------------

		Agent agent = null;

		switch(agentChoice) {
			case 0 : { // PGPE
				agent = new PGPE(pol, gamma, nEpiPerLearningStep,
							ArrUtils.constvec(pol.getParamsSize(),0.01), 0.01,
							new ConstantStepSize(1.0));
				break;
			}
			case 1 : { // CEPS
				final int nPolEvalPerUpdate = nEpiPerLearningStep;
				final int nEpiPerPolEval = 1;
				// must have nEpiPerPolEval * nPolEvalPerUpdate == nEpiPerLearningStep
				final int nbSelectedPol = (int) Math.ceil(nPolEvalPerUpdate*10./100.);
				agent = new CEPS(pol, nPolEvalPerUpdate,
							nEpiPerPolEval,	nbSelectedPol, gamma,
							ArrUtils.constvec(pol.getParamsSize(),0.1), 1e-2, false, false);
				break;
			}
			case 2 : { // REINFORCE
				final ConstantStepSize stepSize = new ConstantStepSize(0.001);
				//final DecreasingStepSize stepSize = new DecreasingStepSize(0.01,1000);
				agent = new REINFORCE((ILogDifferentiablePolicy)pol, xDim, uDim,
							stepSize, nEpiPerLearningStep, gamma);
				break;
			}
			case 3 : { // BasicAC
				agent = new BasicAC((ILogDifferentiablePolicy)pol, td, actorStepSize);
				break;
			}
			case 4 : { // VAC
				agent = new VAC((ILogDifferentiablePolicy)pol, td, actorStepSize, xDim, uDim);
				break;
			}
			case 5 : { // TDNAC
				DiscountFactor kappa = new DiscountFactor(1);
				agent = new TDNAC((ILogDifferentiablePolicy)pol, td, actorStepSize,
							kappa, xDim, uDim);
				break;
			}
			case 6 : { // NAC
				final DiscountFactor lambda = new DiscountFactor(0.5);
				final DiscountFactor kappa = new DiscountFactor(1);
				final ConstantStepSize nacStepSize = new ConstantStepSize(1.0);
				agent = new NAC((ILogDifferentiablePolicy)pol, stateFeat, nacStepSize,
						gamma, lambda, kappa, maxT*nEpiPerLearningStep,
						0.001, xDim, uDim );
				aFunction = ((NAC)agent).getAFunction(); //TODO necessary ?
				vFunction = ((NAC)agent).getVFunction();
				break;
			}
		}
		if(!(agent instanceof EnvironmentListener)) {
			throw new RuntimeException("agent must be an EnvironmentListener");
		}
		env.addListener((EnvironmentListener)agent); // to learn, he must be listening to his environment !

		//---[ Create a logger that will store the sample trajectories ]--------
		final Logger log = new Logger(xDim,uDim);
		env.addListener(log); // make it listen to the environment

		// Plot the mean action from the barycentric interpolation policy
		//pol.plot( stateSampleGrid )
		// Plot the initial state-value approximation
		//td.vFunction.plot( stateSampleGrid )

		// For plotting
		PolicyPlot polPlot = null;
		if(dispPolicy) {
			polPlot = new PolicyPlot(pol, stateSampleGrid, uMin, uMax);
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

//		TrajectoriesPlot trajPlot = new TrajectoriesPlot(
//				"Learning trajectories", xDim, uDim, 0,
//				xBounds, uBounds, null, maxT);

		//---[ Learning loop ]--------------------------------------------------
		System.out.println( "Learning with "+agent.toString() );
		for(int i=0; i<nLearningStep; i++) {

			// Learn during nEpiPerLearningStep episodes
		    log.reset(); // clear the testing history
		    env.interact(agent,nEpiPerLearningStep,maxT);

		    // Plotting
		    //qPlot.plot();
		    //vPlot.plot();
		    //trajPlot.plot(log.getEpisodes());

		    // Testing
		    log.reset(); // clear the learning history
		    env.removeListener((EnvironmentListener)agent); // won't learn during testing

		    // Prepare the policy for testing : make it deterministic
		    if(pol instanceof SharedParamsLGPolicy) {
		        ((SharedParamsLGPolicy)pol).setSigma(ArrUtils.zeros(uDim));
		    }
		    //if(pol instanceof LinearGaussianPolicyWithSDNoise)
		    //    pol.setSDNoiseStdDev(0.*SDNoiseStdDev);

		    // Test on nEpiPerTestStep episodes
		    env.interact(agent,nEpiPerTestStep,maxT);

		    // Plotting and printing some stats
		    //System.out.println( log );
		    System.out.println("nepi=" + (i+1)*nEpiPerLearningStep
		          + " J=" + ArrUtils.mean(log.discountedReward(gamma)));
		    //System.out.println("theta="+Utils.vectorToString(pol.getTheta()));
		    if(dispPolicy) {
		    	polPlot.plot();
		    }

		    // Prepare the policy for learning : restore the stochasticity
		    if(pol instanceof SharedParamsLGPolicy) {
		    	((SharedParamsLGPolicy)pol).setSigma(noiseStdDev);
		    }
		    //if(pol instanceof LinearGaussianPolicyWithSDNoise)
		    //    pol.setSDNoiseStdDev(SDNoiseStdDev);
		    env.addListener((EnvironmentListener)agent);
		}
	}

}
