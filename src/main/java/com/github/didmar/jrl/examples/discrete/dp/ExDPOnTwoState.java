package com.github.didmar.jrl.examples.discrete.dp;

import java.io.IOException;

import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.TabularQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.TabularVFunction;
import com.github.didmar.jrl.features.TabularStateActionFeatures;
import com.github.didmar.jrl.mdp.DiscreteMDP;
import com.github.didmar.jrl.mdp.TwoStateMDP;
import com.github.didmar.jrl.mdp.dp.LSPE;
import com.github.didmar.jrl.mdp.dp.PolicyEvaluation;
import com.github.didmar.jrl.mdp.dp.PolicyIteration;
import com.github.didmar.jrl.mdp.dp.ValueIteration;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.QFunctionPlot;
import com.github.didmar.jrl.utils.plot.VFunctionPlot;

/**
 * Testing the Two State MDP.
 * 
 * @author Didier Marin
 */
public class ExDPOnTwoState {

	public static void main(String[] args) throws Exception {
		final TwoStateMDP mdp = new TwoStateMDP();
		//final DiscreteMDPEnvironment env = new DiscreteMDPEnvironment(mdp);
		final DiscountFactor gamma = new DiscountFactor(0.95);
		
		final double[][] states = mdp.statesGrid();
		final double[][] actions = mdp.actionsGrid();
		
		//---[ Evaluating a policy ]--------------------------------------------
		
		final double[][] pol = ArrUtils.constmat(mdp.n,mdp.m,1./((double)mdp.m));
		
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
		System.out.println("PE Q="+ArrUtils.toString(peQ));
		double peQBellmanError = mdp.QBellmanError(peQ, pol, gamma);
		System.out.println("err="+peQBellmanError);
		
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
		double[][] lspeQ = new double[mdp.n][mdp.m];
		for (int x = 0; x < mdp.n; x++) {
			for (int u = 0; u < mdp.m; u++) {
				lspeQ[x][u] = lspeQFunction.get(new double[]{x}, new double[]{u});
			}
		}
		System.out.println("LSPE Q="+ArrUtils.toString(lspeQ));
		double lspeQBellmanError = mdp.QBellmanError(lspeQ, pol, gamma);
		System.out.println("err="+lspeQBellmanError);
		System.out.println("-------------------------------------------------");
		
		//---[ Computing the optimal policy ]-----------------------------------
		
		// 1) Value Iteration
		
		final int viMaxIter = 100000;
		final double epsilon = 0.00000001;
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
