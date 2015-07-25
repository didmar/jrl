package com.github.didmar.jrl.mdp;

import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * The Hall is a simple uncontrolled MDP used in Baird 1995 "Residual 
 * Algorithms: Reinforcement Learning with Function Approximation". 
 * @author Didier Marin
 */
public final class HallMDP extends DiscreteMDP {

	public HallMDP(int n) {
		super(n, 1);
		ArrUtils.zeros(P0);
		P0[0] = 1.;
		ArrUtils.zeros(R);
		R[n-1][0] = 1.;
		for(int x=0; x<n-1; x++) {
			ArrUtils.zeros(P[x][0]);
			P[x][0][x+1] = 1.;
		}
		P[n-1][0][n-1] = 1.;
	}
}
