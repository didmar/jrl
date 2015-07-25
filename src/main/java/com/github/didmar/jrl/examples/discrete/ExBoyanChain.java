package com.github.didmar.jrl.examples.discrete;

import java.io.IOException;

import com.github.didmar.jrl.agent.PolicyAgent;
import com.github.didmar.jrl.environment.discrete.BoyanChain;
import com.github.didmar.jrl.evaluation.valuefunction.BoyanChainVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.ParametricVFunction;
import com.github.didmar.jrl.evaluation.vflearner.gtd.TDC;
import com.github.didmar.jrl.evaluation.vflearner.ktd.KTDAV;
import com.github.didmar.jrl.evaluation.vflearner.ktd.KTDLambda;
import com.github.didmar.jrl.evaluation.vflearner.ktd.KTDZero;
import com.github.didmar.jrl.evaluation.vflearner.ktd.LinearKTDZero;
import com.github.didmar.jrl.evaluation.vflearner.lstd.ILSTD;
import com.github.didmar.jrl.evaluation.vflearner.lstd.LSTD;
import com.github.didmar.jrl.evaluation.vflearner.td.TDLambda;
import com.github.didmar.jrl.evaluation.vflearner.td.TDZero;
import com.github.didmar.jrl.features.BoyanChainFeatures;
import com.github.didmar.jrl.policy.ConstantActionPolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.stepsize.DecreasingStepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.plot.Plot2D;

/**
 * Testing the evaluation of the Boyan Chain.
 * 
 * @author Didier Marin
 */
@SuppressWarnings("unused")
public class ExBoyanChain {

	public static void main(String[] args) throws IOException {
		
		final BoyanChain env = new BoyanChain();
		// Linear approximation of the state value function,
		// using Boyan Chain-specific features
		final BoyanChainFeatures feat = new BoyanChainFeatures();
		LinearVFunction vFunction = new LinearVFunction(feat);
		// The true state value function for the Boyan Chain
		final BoyanChainVFunction optVFunction = new BoyanChainVFunction();
		
		final DiscountFactor gamma = new DiscountFactor(1);
		final int nbIterations = 1000;
		final int nbEpisodesPerIter = 10;
		final int maxT = BoyanChain.CHAIN_LENGTH;
		
		final ConstantStepSize stepSize = new ConstantStepSize(0.01);
		//final DecreasingStepSize stepSize = new DecreasingStepSize(1.0,100);
		//TDZero tdLearner = new TDZero(vFunction, stepSize, gamma);
		//TDLambda tdLearner = new TDLambda(vFunction, stepSize, gamma, new DiscountFactor(0));
		//TDC tdLearner = new TDC(vFunction, gamma, stepSize, 0.0);
		//LSTD tdLearner = new LSTD(vFunction, gamma, nbIterations*nbEpisodesPerIter, 0.);
		//ILSTD tdLearner = new ILSTD(vFunction, gamma, new DiscountFactor(0), 1, 0.1);
		//KTDZero tdLearner = new KTDZero(vFunction,gamma,0.1,0.1,0.1,0.001);
		//LinearKTDZero tdLearner = new LinearKTDZero(vFunction,gamma,0.1,0.1,0.1);
		KTDLambda tdLearner = new KTDLambda(vFunction, gamma,
											new DiscountFactor(0.01),
											0.1, 1e-1, 0.1, 0.001, 1e-2);
		env.addListener(tdLearner);
		
		final double[] u = {0.};
		final ConstantActionPolicy pol = new ConstantActionPolicy(u);
		final PolicyAgent agent = new PolicyAgent(pol);
		
		// Used to store the state when computing the approximation error
		final double[] x = {0.};
		
		final double[] iterations = new double[nbIterations];
		final double[] MSEs = new double[nbIterations];
		for(int i=0; i<nbIterations; i++) {
			iterations[i] = i;
			env.interact(agent, nbEpisodesPerIter, maxT);
			double error = 0.;
			for(int j=0; j<BoyanChain.CHAIN_LENGTH; j++) {
				x[0] = j;
				error += Math.pow(vFunction.get(x)-optVFunction.get(x), 2);
			}
			System.out.println("Episode "+(i+1)+" : MSE="+error);
			MSEs[i] = error;
		}
		
		for(int i=0; i<BoyanChain.CHAIN_LENGTH; i++) {
			x[0] = i;
			System.out.println("V("+i+") = "
					+ vFunction.get(x)
					+ " (optimal : " + optVFunction.get(x) + ")");
		}
		
		Plot2D plt = new Plot2D("State-value approximation error",
				"iterations","MSE");
		plt.plot(iterations, MSEs);
		//plt.plot(new double[]{1,2,3}, new double[]{10.,5.,1.});
		
		System.out.println("Press a key to terminate...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
