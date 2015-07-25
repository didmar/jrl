package com.github.didmar.jrl.examples.discrete;

import com.github.didmar.jrl.agent.PolicyAgent;
import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.environment.discrete.DiscreteMDPEnvironment;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.vflearner.VFunctionLearner;
import com.github.didmar.jrl.evaluation.vflearner.gtd.TDC;
import com.github.didmar.jrl.evaluation.vflearner.ktd.KTDLambda;
import com.github.didmar.jrl.evaluation.vflearner.ktd.KTDZero;
import com.github.didmar.jrl.evaluation.vflearner.ktd.LinearKTDZero;
import com.github.didmar.jrl.evaluation.vflearner.lstd.ILSTD;
import com.github.didmar.jrl.evaluation.vflearner.lstd.LSTD;
import com.github.didmar.jrl.evaluation.vflearner.td.TDLambda;
import com.github.didmar.jrl.evaluation.vflearner.td.TDZero;
import com.github.didmar.jrl.features.BairdStarFeatures;
import com.github.didmar.jrl.mdp.BairdStarMDP;
import com.github.didmar.jrl.policy.ConstantActionPolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.RandUtils;

/**
 * Testing Baird's off-policy counterexample.
 * 
 * @see com.github.didmar.jrl.features.BairdStarFeatures
 * @author Didier Marin
 */
public class ExBairdStar {

	public static void main(String[] args) {
		
		//---[ Create a Baird star environment ]--------------------------------
		final BairdStarMDP mdp = new BairdStarMDP();
		final DiscreteMDPEnvironment env = new DiscreteMDPEnvironment(mdp);
		
		//---[ Set the run parameters ]-----------------------------------------
		final DiscountFactor gamma = new DiscountFactor(0.95);
		final int maxT = 100; // maximum episode duration
		final int nbIterations = 1000;
		final int nbEpisodesPerIter = 1;
		
		//---[ Create features for value approx. and policy ]-------------------

		final BairdStarFeatures stateFeat
			= new BairdStarFeatures();
		
		//---[ Create a state-action value function approximator ]--------------
		
		double[] params = new double[stateFeat.outDim];
		for (int i = 0; i < params.length; i++) {
			params[i] = RandUtils.nextDouble();
		}
		LinearVFunction vFunction = new LinearVFunction(stateFeat,params);
		
		//---[ Create a value function learner ]--------------------------------
		
		final ConstantStepSize stepSize = new ConstantStepSize(0.1);
		try {
			env.addListener(new TDZero((LinearVFunction)vFunction.clone(), stepSize, gamma));
			env.addListener(new TDLambda((LinearVFunction)vFunction.clone(), stepSize, gamma, new DiscountFactor(0)));
			env.addListener(new TDC((LinearVFunction)vFunction.clone(), gamma, stepSize, 0.));
			env.addListener(new LSTD((LinearVFunction)vFunction.clone(), gamma, nbIterations*nbEpisodesPerIter, 1.));
			env.addListener(new ILSTD((LinearVFunction)vFunction.clone(), gamma, new DiscountFactor(0), 1, 1.));
			env.addListener(new KTDZero((LinearVFunction)vFunction.clone(),gamma,0.1,0.1,0.1,0.001));
			env.addListener(new LinearKTDZero((LinearVFunction)vFunction.clone(),gamma,0.1,0.1,0.1));
			env.addListener(new KTDLambda((LinearVFunction)vFunction.clone(), gamma, new DiscountFactor(0.4), 0.1, 1e-1,
					0.1, 0.001, 1e-2));
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		
		//---[ Create a dummy agent ]-------------------------------------------
		
		final double[] u = {0.};
		ConstantActionPolicy pol = new ConstantActionPolicy(u);
		PolicyAgent agent = new PolicyAgent(pol);
		
		for(int i=0; i<nbIterations; i++) {
			// Perform some interactions to feed the learners 
			env.interact(agent, nbEpisodesPerIter, maxT);
			// Evaluate the estimated value function of each learner
			System.out.print("Epi "+(i+1)+" MSE = ");
			for(EnvironmentListener l : env) {
				LinearVFunction V = (LinearVFunction) ((VFunctionLearner)l).getVFunction();
				double error = 0.;
				params = V.getParams();
				for(int j=0; j<params.length; j++) {
					error += Math.pow(params[j], 2);
				}
				System.out.print(l.toString()+":"+error+" ");
			}
			System.out.println();
		}
	}
}
