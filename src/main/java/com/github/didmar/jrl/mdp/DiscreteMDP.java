package com.github.didmar.jrl.mdp;

import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Markov Decision Process with finite discrete states and actions, and a
 * deterministic reward function.
 *
 * @author Didier Marin
 */
public class DiscreteMDP {

	/** Initial state probability table */
	public final double[] P0;
	/** Transition probability table */
	public final double[][][] P;
	/** Reward table */
	public final double[][] R;
	/** Number of states */
	public final int n;
	/** Number of actions */
	public final int m;

	public DiscreteMDP(double[] P0, double[][][] P,	double[][] R) {
		if(P0.length != P.length) {
			throw new IllegalArgumentException(
					"P0 length is not consistent with P first dimension");
		}
		n = P0.length;
		m = P[0].length;
		if(!ArrUtils.hasShape(P, n, m, n)) {
			throw new IllegalArgumentException(
					"P shape is not consistent");
		}
		if(!ArrUtils.hasShape(R, n, m)) {
			throw new IllegalArgumentException(
					"R shape is not consistent");
		}
		this.P0 = P0;
		this.P = P;
		this.R = R;

	}

	public DiscreteMDP(int n, int m) {
		this.n = n;
		this.m = m;
		this.P0 = new double[n];
		this.P = new double[n][m][n];
		this.R = new double[n][m];
	}

	/** Compute the transition kernel for a given policy */
	public final void computeTransitionKernel(double[][] pol, double[][] K) {
		for(int x=0; x<n; x++) {
			for(int xn=0; xn<n; xn++) {
				K[x][xn] = 0.;
				for(int u=0; u<m; u++) {
					K[x][xn] += pol[x][u] * P[x][u][xn];
				}
			}
		}
	}

	/** Compute the state-action value function from the state value function
	 * of given policy */
	public final void computeQfromV(double[] V, DiscountFactor gamma, double[][] Q) {
		for(int x=0; x<n; x++) {
	    	for(int u=0; u<m; u++) {
	            Q[x][u] = R[x][u] + gamma.value * ArrUtils.dotProduct(P[x][u], V, n);
	    	}
	    }
	}

	/** Compute the greedy policy with respect to a given state action value
	 * function */
	public final void computeGreedyPolicy(double[][] Q, int[] pol) {
		for(int x=0; x<n; x++) {
	    	pol[x] = ArrUtils.argmax(Q[x]);
	    }
	}

	/** Compute the expected discounted reward given a state value function */
	public final double expectedDiscountedReward(double[] V) {
		return ArrUtils.dotProduct(V, P0, n);
	}

	/**
	 * Check that the transition matrix P sums to 1 over next states,
	 * for each state-action pair.
	 * @param P	a transition matrix
	 * @return true is the transition matrix is valid, false otherwise
	 */
	public static final boolean verifyP(double[][][] P) {
		for (int x = 0; x < P.length; x++) {
			for (int u = 0; u < P[x].length; u++) {
				if(ArrUtils.sum(P[x][u]) != 1.) {
					return false;
				}
			}
		}
		return true;
	}

	public final double[][] statesGrid() {
		return ArrUtils.buildGrid(ArrUtils.zeros(1),
				               ArrUtils.constvec(1,n-1),
				               n);
	}

	public final double[][] actionsGrid() {
		return ArrUtils.buildGrid(ArrUtils.zeros(1),
				               ArrUtils.constvec(1,m-1),
				               m);
	}

	public final double QBellmanError(double[][] Q, double[][] pol,
			DiscountFactor gamma) {
		double error = 0.;
		for(int x=0; x<n; x++) {
			for(int u=0; u<m; u++) {
				double Qxnun = 0.;
				for(int xn=0; xn<n; xn++) {
					for(int un=0; un<m; un++) {
						Qxnun += P[x][u][xn] * pol[xn][un] * Q[xn][un];
					}
				}
				error += Math.pow(R[x][u]+gamma.value*Qxnun-Q[x][u], 2);
			}
		}
		return error;
	}
}
