package com.github.didmar.jrl.tests;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

import com.github.didmar.jrl.agent.Agent;
import com.github.didmar.jrl.agent.PolicyAgent;
import com.github.didmar.jrl.environment.Environment;
import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.environment.discrete.BoyanChain;
import com.github.didmar.jrl.environment.discrete.DiscreteEnvironment;
import com.github.didmar.jrl.environment.discrete.DiscreteMDPEnvironment;
import com.github.didmar.jrl.evaluation.valuefunction.BoyanChainVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.TabularQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.TabularVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.VFunction;
import com.github.didmar.jrl.evaluation.vflearner.VFunctionLearner;
import com.github.didmar.jrl.evaluation.vflearner.gtd.TDC;
import com.github.didmar.jrl.evaluation.vflearner.ktd.KTDLambda;
import com.github.didmar.jrl.evaluation.vflearner.ktd.LinearKTDZero;
import com.github.didmar.jrl.evaluation.vflearner.lstd.ILSTD;
import com.github.didmar.jrl.evaluation.vflearner.lstd.LSTD;
import com.github.didmar.jrl.evaluation.vflearner.td.TDLambda;
import com.github.didmar.jrl.evaluation.vflearner.td.TDZero;
import com.github.didmar.jrl.features.BoyanChainFeatures;
import com.github.didmar.jrl.features.TabularStateActionFeatures;
import com.github.didmar.jrl.features.TabularStateFeatures;
import com.github.didmar.jrl.mdp.TwoStateMDP;
import com.github.didmar.jrl.mdp.dp.PolicyEvaluation;
import com.github.didmar.jrl.policy.BoltzmannPolicyOverQ;
import com.github.didmar.jrl.policy.ConstantActionPolicy;
import com.github.didmar.jrl.policy.DiscreteRandomPolicy;
import com.github.didmar.jrl.policy.EpsGreedyPolicyOverQ;
import com.github.didmar.jrl.policy.Policy;
import com.github.didmar.jrl.policy.QFunctionBasedPolicy;
import com.github.didmar.jrl.policy.UniRandomPolicy;
import com.github.didmar.jrl.stepsize.ConstantStepSize;
import com.github.didmar.jrl.stepsize.DecreasingStepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.QFunctionPlot;

public class TestTDAlgorithms {

	@Test
	public void boyanChain() {
		
		System.out.println("Testing on BoyanChain");
		
		final BoyanChain env = new BoyanChain();
		final BoyanChainFeatures feat = new BoyanChainFeatures();
		final BoyanChainVFunction trueVFunction = new BoyanChainVFunction();
		final DiscountFactor gamma = new DiscountFactor(0.01);
		final int maxT = BoyanChain.CHAIN_LENGTH;
		final ConstantActionPolicy pol = new ConstantActionPolicy(new double[]{0});
		final PolicyAgent agent = new PolicyAgent(pol);
		
		final int nbEpisodes = 100;
		final double targetMSE = 1800;
		
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new TDZero(new LinearVFunction(feat),
						   new DecreasingStepSize(1.0,100),
						   gamma)
				));
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new TDLambda(new LinearVFunction(feat),
						new DecreasingStepSize(1.0,100), gamma,
						new DiscountFactor(0.01))
				));
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new TDC(new LinearVFunction(feat), gamma,
						new DecreasingStepSize(1.0,100), 0.0)
				));
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new LSTD(new LinearVFunction(feat), gamma, nbEpisodes, 0.)
				));
		/*assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, optVFunction,
				new ILSTD(new LinearVFunction(feat), gamma,
						  new DiscountFactor(0), 1, 1.)
				));*/
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new LinearKTDZero(new LinearVFunction(feat),gamma,0.1,0.1,0.1)
				));
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new KTDLambda(new LinearVFunction(feat), gamma,
						new DiscountFactor(0.4), 0.1, 1e-1,
						0.1, 0.001, 1e-2)
				));		
	}
	
	@Test
	public void twoState() {
		
		System.out.println("Testing on TwoState");
		
		final TwoStateMDP mdp = new TwoStateMDP();
		final DiscreteMDPEnvironment env = new DiscreteMDPEnvironment(mdp);
		final DiscountFactor gamma = new DiscountFactor(0.95);
		final int maxT = 1000;
		final TabularStateFeatures feat	= new TabularStateFeatures(mdp);
		final double[][] actions = mdp.actionsGrid();
		final Policy pol = new DiscreteRandomPolicy(actions);
		
		// Compute the true value function using policy evaluation 
		final double[][] polTab = ArrUtils.constmat(mdp.n,mdp.m,1./((double)mdp.m));
		PolicyEvaluation pe = new PolicyEvaluation(mdp, polTab, gamma);
		double[][] peQ = pe.getQ();
		double[] peV = ArrUtils.zeros(mdp.n);
		for(int x=0; x < mdp.n; x++) {
			for(int u=0; u < mdp.m; u++) {
				peV[x] += peQ[x][u] * polTab[x][u];
			}
			//System.out.println("trueVFunction["+x+"]="+peV[x]);
		}
		TabularVFunction trueVFunction = new TabularVFunction(peV);
		
		final int nbEpisodes = 1000;
		final double targetMSE = 100.0;
		final Agent agent = new PolicyAgent(pol);
		
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new TDZero(new LinearVFunction(feat),
						   new DecreasingStepSize(1.0,100),
						   gamma)
				));
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new TDLambda(new LinearVFunction(feat),
						new DecreasingStepSize(1.0,100), gamma,
						new DiscountFactor(0.01))
				));
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new TDC(new LinearVFunction(feat), gamma,
						new DecreasingStepSize(1.0,100), 0.0)
				));
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new LSTD(new LinearVFunction(feat),
						gamma, nbEpisodes, 0.0001)
				));
		/*assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, optVFunction,
				new ILSTD(new LinearVFunction(feat), gamma,
						  new DiscountFactor(0), 1, 1.)
				));*/
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new LinearKTDZero(new LinearVFunction(feat),gamma,
						10.0,0.001,0.5)
				));
		assertTrue(MSELessThan( targetMSE,
				env, nbEpisodes,
				maxT, agent, trueVFunction,
				new KTDLambda(new LinearVFunction(feat), gamma,
						new DiscountFactor(0.01), 0.1, 0.1,
						0.1, 0.1, 0.1)
				));		
	}
	
	private boolean MSELessThan(double targetMSE,
			DiscreteEnvironment env,
			int nbEpisodes, int maxT,
			Agent agent, VFunction trueVFunction,
			VFunctionLearner learner) {
		
		env.addListener((EnvironmentListener)learner);
		env.interact(agent, nbEpisodes, maxT);
		env.removeListener((EnvironmentListener)learner);
		
		final double[] x = {0.};
		double error = 0.;
		VFunction vFunction = learner.getVFunction();
		for(int j=0; j<env.getXCard(); j++) {
			x[0] = j;
			error += Math.pow(vFunction.get(x)-trueVFunction.get(x), 2);
		}
		System.out.println(learner.toString()+" : error="+error);
		return (error <= targetMSE);
	}

}
