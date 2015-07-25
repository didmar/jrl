package com.github.didmar.jrl.examples.continuous;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

import com.github.didmar.jrl.agent.CEPS;
import com.github.didmar.jrl.agent.LearningAgent;
import com.github.didmar.jrl.agent.PGPE;
import com.github.didmar.jrl.agent.ac.BasicAC;
import com.github.didmar.jrl.agent.ac.TDNAC;
import com.github.didmar.jrl.agent.ac.VAC;
import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.environment.PointMass;
import com.github.didmar.jrl.environment.dynsys.DynPointMass;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.vflearner.td.TDZero;
import com.github.didmar.jrl.features.CompatibleFeatures;
import com.github.didmar.jrl.features.RBFFeatures;
import com.github.didmar.jrl.policy.LinearGaussianPolicy;
import com.github.didmar.jrl.stepsize.DecreasingStepSize;
import com.github.didmar.jrl.utils.CEParametersDistribution;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.PGPEParametersDistribution;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.PolicyPlot;
import com.github.didmar.jrl.utils.plot.QFunctionPlot;
import com.github.didmar.jrl.utils.plot.TrajectoriesPlot;
import com.github.didmar.jrl.utils.plot.VFunctionPlot;

/**
 * Learn an optimal policy on the Dynamic Point Mass problem
 * @author Didier Marin
 */
@SuppressWarnings("unused")
public class ExDynPointMass {

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		int nbTrials = 1;
		if(args.length > 0) {
			nbTrials = Integer.parseInt(args[0]);
		}
		String outputPath = ".";
		for(int i=0; i<nbTrials; i++) {
			System.out.println("trial "+(i+1)+"/"+nbTrials);
			performTest(i+1, outputPath);
		}
	}

	/**
	 *
	 * @param trial
	 * @param outputPath
	 * @return the name of the result file
	 * @throws Exception
	 */
	public static String performTest(int trial, String outputPath) throws Exception {

		String resultFilename = "perf_dpm_";

		//---[ Create a dynamic point mass environment ]------------------------
		final double[] x0 = new double[]{0.25, 0.};
		final double targetPos = 0.75;
		final double dt = 0.01;
		final double maxSpeed = 1.;
		final boolean goalIsTerminal = true;
		final double targetRadius   = 0.01;
		final double targetMaxSpeed = 0.1;
		final double goalReward = 1.0;
		final double costFactor = 0.01;
		final boolean randomStartState = false;
		final DynPointMass env = new DynPointMass(x0, targetPos,
				dt, maxSpeed, goalIsTerminal, targetRadius, targetMaxSpeed, goalReward,
				costFactor, randomStartState);
		final int xDim = env.getXDim();
		final int uDim = env.getUDim();
		final double[] xMin = env.getXMin();
		final double[] xMax = env.getXMax();
		final double[] uMin = new double[]{-1.};
		final double[] uMax = new double[]{+1.};

		final double[][] stateSampleGrid = ArrUtils.buildGrid(xMin,xMax,101);
		final double[][] actionSampleGrid = ArrUtils.buildGrid(uMin,uMax,101);

		//---[ Set the run parameters ]-----------------------------------------
		final DiscountFactor gamma = new DiscountFactor(0.99);
		final int maxT = 2000; // maximum episode duration
		final int nLearningStep = 46; // => 100000 learning episodes in total
		final int nEpiPerTestStep = 1;

		//---[ Create a logger that will store the sample trajectories ]--------
		final Logger log = new Logger(xDim,uDim);
		env.addListener(log); // make it listen to the environment

		//---[ Create features for value approx. and policy ]-------
		final int nbStateFeat = 21;
		final double[][] stateCenters = ArrUtils.buildGrid(xMin,xMax,nbStateFeat);
		final double[] sigma = new double[]{0.05,0.1}; //Utils.constvec(xDim,0.01);
		final RBFFeatures stateFeat = new RBFFeatures(stateCenters, sigma, false);
		final RBFFeatures polStateFeat = new RBFFeatures(stateCenters, sigma, true);

		//---[ Choose a policy ]------------------------------------------------
		// - Linear Gaussian with constant noise
		final double[] noiseStdDev = ArrUtils.zeros(uDim); // For FD, iFD, CEPS
//		final double[] noiseStdDev = Utils.constvec(uDim,0.3); // For the rest
		LinearGaussianPolicy pol = new LinearGaussianPolicy(polStateFeat, noiseStdDev,
				uMin, uMax, true);
		// - Linear Gaussian with state dependent noise
		//SDNoiseStdDev = sqrt(0.1)*ones(uDim)
		//pol = LinearGaussianPolicyWithSDNoise(polStateFeat, SDNoiseStdDev, \
		//                                      uBounds, redrawNoiseForEachAction=True)
		//env.addListener(pol) # to renew the SD noise at the begining of each episode

		//---[ Create a Value function approximator ]---------------------------
		LinearVFunction vFunction = new LinearVFunction(stateFeat);
		LinearQFunction aFunction = new LinearQFunction(
				new CompatibleFeatures(pol, xDim, uDim), xDim, uDim);

		//System.out.println("Debug : "
		//		+Utils.toString(aFunction.getFeatures().phi(new double[]{0.5,0.1,0.0})));

		//---[ Create a TD Learner (for BasicAC, iNAC, VAC, ...) ]--------------

		// Create a step-size for TD
		//final ConstantStepSize TDstepSize = new ConstantStepSize(0.0001);
		//final DecreasingStepSize TDstepSize = new DecreasingStepSize(0.01,1000,2./3.);

		//final DecreasingStepSize[] twoTimescaleStepsSizes =
		//	DecreasingStepSize.createTwoTimescaleStepSizes(0.01,1e5,0.001,1e5);

		final DecreasingStepSize[] twoTimescaleStepsSizes =
			DecreasingStepSize.createTwoTimescaleStepSizes(0.01,1e8,0.001,1e8);

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

		// +++ CEPS
		final int nPolEvalPerUpdate = 100;
		final int nEpiPerPolEval = 1;
		final int nbSelectedPol = (int) Math.ceil(nPolEvalPerUpdate*50./100.);
		final LearningAgent agent = new CEPS(pol, nPolEvalPerUpdate,
				nEpiPerPolEval,	nbSelectedPol, gamma,
				ArrUtils.constvec(pol.getParamsSize(),1.), 1e-2, false, false);

		// +++ CMAESPS
//		final int nPolEvalPerUpdate = nEpiPerLearningStep;
//		final int nEpiPerPolEval = 1;
//		assert(nEpiPerPolEval * nPolEvalPerUpdate == nEpiPerLearningStep);
//		final int nbSelectedPol = (int) Math.ceil(nPolEvalPerUpdate*10./100.);
//		final LearningAgent agent = new CMAESPS(pol, xDim, uDim, nPolEvalPerUpdate,
//				nEpiPerPolEval,	gamma,
//				Utils.constvec(pol.getParamsSize(),0.1));

		// +++ BasicAC
//		final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
//		final LearningAgent agent = new BasicAC(pol, td, actorStepSize);

		// +++ VAC
//		final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
//		final LearningAgent agent = new VAC(pol, td, actorStepSize, xDim, uDim);
//		resultFilename += "VAC_alpha_"+actorStepSize.getAlpha0()
//			+"_"+actorStepSize.getAlphaC()+"_beta_"
//			+twoTimescaleStepsSizes[0].getAlpha0()+"_"
//			+twoTimescaleStepsSizes[0].getAlphaC()+"_";

		// +++ TDNAC
//		final DecreasingStepSize actorStepSize = twoTimescaleStepsSizes[1];
//		final double kappa = 0.999;
//		final LearningAgent agent = new TDNAC(pol, td, actorStepSize, kappa, xDim, uDim);
//		resultFilename += "TDNAC_kappa_"+kappa+"_alpha_"+actorStepSize.getAlpha0()
//			+"_"+actorStepSize.getAlphaC()+"_beta_"
//			+twoTimescaleStepsSizes[0].getAlpha0()+"_"
//			+twoTimescaleStepsSizes[0].getAlphaC()+"_";

		// +++ NAC
//		final ConstantStepSize nacStepSize = new ConstantStepSize(1.0);
//		final LearningAgent agent = new NAC(pol, stateFeat, nacStepSize, gamma,
//				0.5, 1., maxT*nEpiPerLearningStep, 0.001, xDim, uDim );
//		aFunction = agent.getAFunction();
//		vFunction = agent.getVFunction();

		env.addListener(agent); // to learn, he must be listening to his environment !

		resultFilename += "trial"+trial;

		//---[ Plotting stuff ]-------------------------------------------------

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

		TrajectoriesPlot trajPlot = new TrajectoriesPlot("Testing trajectories",
				xDim, uDim, 0, xMin, xMax, uMin, uMax, null, maxT);

		//---[ Open the result file ]-------------------------------------------
		File resultFile = new File(outputPath+"/"+resultFilename);
        FileWriter resultFstream = null;
		try {
			resultFstream = new FileWriter(resultFile);
		} catch (IOException e) {
			e.printStackTrace();
		}
        BufferedWriter resultOut = new BufferedWriter(resultFstream);

        //---[ Learning loop ]--------------------------------------------------
		System.out.println( "Learning with "+agent.toString() );
		int nTotalLearningEpi = 0;
		for(int i=18; i<nLearningStep; i++) {

			int nLearningEpi = (((i%9)+1) * ((int)Math.pow(10,i/9))) - nTotalLearningEpi;
			nTotalLearningEpi += nLearningEpi;

			// Learn during nEpiPerLearningStep episodes
		    log.reset(); // clear the testing history
		    env.interact(agent,nLearningEpi,maxT);
//		    qPlot.plot();
//		    vPlot.plot();

		    // Testing
		    log.reset(); // clear the learning history
		    env.removeListener(agent); // won't learn during testing

		    // Prepare the policy for testing : make it deterministic
		    if(pol instanceof LinearGaussianPolicy) {
		        pol.setSigma(ArrUtils.zeros(uDim));
		    }
		    //if isinstance(pol,LinearGaussianPolicyWithSDNoise):
		    //    pol.setSDNoiseStdDev(0.*SDNoiseStdDev)

		    // Test on nEpiPerTestStep episodes
		    env.interact(agent,nEpiPerTestStep,maxT);

		    // Print some stats
		    System.out.print("nepi=" + nTotalLearningEpi
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

//		    polPlot.plot();

		    // FIXME does not show up !!
		    trajPlot.plot(log.getEpisodes());

		    resultOut.write(nTotalLearningEpi + " "
		    		+ ArrUtils.mean(log.discountedReward(gamma)) + "\n");

		    // Prepare the policy for learning : restore the stochasticity
		    if(pol instanceof LinearGaussianPolicy) {
		        pol.setSigma(noiseStdDev);
		    }
		    //if isinstance(pol,LinearGaussianPolicyWithSDNoise):
		    //    pol.setSDNoiseStdDev(SDNoiseStdDev)
		    env.addListener(agent);
		}

		resultOut.close();
		resultFstream.close();

		return resultFilename;
	}

}
