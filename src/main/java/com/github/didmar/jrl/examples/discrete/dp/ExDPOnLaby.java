package com.github.didmar.jrl.examples.discrete.dp;

import java.io.IOException;

import com.github.didmar.jrl.mdp.LabyMDP;
import com.github.didmar.jrl.mdp.dp.PolicyIteration;
import com.github.didmar.jrl.mdp.dp.ValueIteration;
import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * Testing Dynamic Programming methods on the Labyrinth MDP.
 * 
 * @see jrl_testing.mdp.LabyMDP
 * @author Didier Marin
 */
public class ExDPOnLaby {

	public static void main(String[] args) throws Exception {
		int width = 5;
		int height = 5;
		final LabyMDP mdp = new LabyMDP(width,height);
		mdp.setReward(1, 2, +1.0);
		mdp.setReward(3, 2, +0.5);
		mdp.setObstacle(0, 1);
		mdp.setObstacle(1, 1);
		mdp.setObstacle(2, 1);
		mdp.setObstacle(2, 2);
		
		final DiscountFactor gamma = new DiscountFactor(0.95);
		
		//---[ Computing the optimal policy ]-----------------------------------
		
		// 1) Value Iteration
		
		final int viMaxIter = 100000;
		final double epsilon = 1e-15;
		ValueIteration vi = new ValueIteration(mdp, gamma, viMaxIter, epsilon);
		//final int[] viPol = vi.getPol();
		final double[] viV = vi.getV();
		final double viJ = mdp.expectedDiscountedReward(viV);
		System.out.print("Value Iteration  : J="+viJ);
		if(vi.hasConverged()) {
			System.out.println(" (converged after "+vi.convergedAfter()+" iterations)");
		} else {
			System.out.println(" (did not converge)");
		}
		
		// 2) Policy Iteration
		
		final int piMaxIter = 1000;
		PolicyIteration pi = new PolicyIteration(mdp, gamma, piMaxIter);
		final int[] piPol = pi.getPol();
		final double[] piV = pi.getV();
		//final double[][] piQ = pi.getQ();
		final double piJ = mdp.expectedDiscountedReward(piV);
		System.out.print("Policy Iteration : J="+piJ);
		if(pi.hasConverged()) {
			System.out.println(" (converged after "+pi.convergedAfter()+" iterations)");
		} else {
			System.out.println(" (did not converge after "+piMaxIter+" iterations)");
		}
		
		//---[ Show the laby and the optimal policy ]--------------------------
		
		System.out.println("Labyrinth :");
		mdp.printLaby();
		System.out.println();
		System.out.println("Optimal policy :");
		mdp.printPolicy(piPol);
		
		System.out.println("Press a key to terminate...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
