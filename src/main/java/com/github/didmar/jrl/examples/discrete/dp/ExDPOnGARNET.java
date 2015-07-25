package com.github.didmar.jrl.examples.discrete.dp;

import java.io.IOException;

import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.TabularQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.TabularVFunction;
import com.github.didmar.jrl.features.TabularStateActionFeatures;
import com.github.didmar.jrl.mdp.GARNETMDP;
import com.github.didmar.jrl.mdp.dp.LSPE;
import com.github.didmar.jrl.mdp.dp.PolicyEvaluation;
import com.github.didmar.jrl.mdp.dp.PolicyIteration;
import com.github.didmar.jrl.mdp.dp.ValueIteration;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.plot.QFunctionPlot;
import com.github.didmar.jrl.utils.plot.VFunctionPlot;

/**
 * Test Dynamic Programming methods on the GARNET MDP.
 * 
 * @see jrl_testing.mdp.GARNETMDP
 * @author Didier Marin
 */
public class ExDPOnGARNET {
	public static void main(String[] args) throws Exception {
		
		final int n = 5;
		final int m = 5;
		final int b = 3;
		GARNETMDP mdp = new GARNETMDP(n,m,b);
		final DiscountFactor gamma = new DiscountFactor(0.95);
		
		final double[][] states = mdp.statesGrid();
		final double[][] actions = mdp.actionsGrid();
		
		//---[ Evaluating a policy ]--------------------------------------------
		
		final double[][] pol = new double[mdp.n][mdp.m];
		for(int x=0; x<mdp.n; x++) {
			for(int u=0; u<mdp.m; u++) {
				pol[x][u] = 1. / ((double)mdp.m);
			}
		}
		
		// 1) Policy Evaluation
		
		PolicyEvaluation pe = new PolicyEvaluation(mdp, pol, gamma);
		double[][] peQ = pe.getQ();
		try {
			QFunctionPlot peQPlot = new QFunctionPlot("Q-function with PE",
					new TabularQFunction(peQ), states, actions);
			peQPlot.plotHistogram();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// 2) Least-Squares Policy Evaluation
		
		TabularStateActionFeatures stateActionFeat
			= new TabularStateActionFeatures(mdp);
		LinearQFunction lspeQFunction
			= new LinearQFunction(stateActionFeat,1,1);
		new LSPE(mdp, pol, lspeQFunction, gamma);
		try {
			QFunctionPlot lspeQPlot = new QFunctionPlot("Q-function with LSPE",
					lspeQFunction, states, actions);
			lspeQPlot.plotHistogram();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//---[ Computing the optimal policy ]-----------------------------------
		
		// 1) Value Iteration
		
		final int viMaxIter = 1000;
		final double epsilon = -1.;
		ValueIteration vi = new ValueIteration(mdp, gamma, viMaxIter, epsilon);
		final int[] viPol = vi.getPol();
		final double[] viV = vi.getV();
		final double viJ = mdp.expectedDiscountedReward(viV);
		System.out.println("Value Iteration  : J="+viJ);
		try {
			VFunctionPlot viVPlot = new VFunctionPlot("Opt V-function with VI",
					new TabularVFunction(viV),
					states);
			viVPlot.plotHistogram();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// 2) Policy Iteration
		
		final int piMaxIter = 10000;
		PolicyIteration pi = new PolicyIteration(mdp, gamma, piMaxIter);
		final int[] piPol = pi.getPol();
		final double[] piV = pi.getV();
		final double[][] piQ = pi.getQ();
		final double piJ = mdp.expectedDiscountedReward(piV);
		System.out.println("Policy Iteration : J="+piJ);
		try {
			QFunctionPlot piQPlot = new QFunctionPlot("Opt Q-function with PI",
					new TabularQFunction(piQ),
					states,
					actions);
			piQPlot.plotHistogram();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//---[ Check some stuff ]-----------------------------------------------
		for(int x=0; x<mdp.n; x++) {
			assert(viPol[x] == piPol[x]);
		}
		
		System.out.println("Press a key to terminate...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
