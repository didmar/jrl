package com.github.didmar.jrl.mdp.dp;

import com.github.didmar.jrl.mdp.DiscreteMDP;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Implements Policy Iteration.
 * @author Didier Marin
 */
public final class ValueIteration {
	
	private final DiscreteMDP mdp;
	/** Discount factor */
	private final DiscountFactor gamma;
	/** Q-Function represented as a matrix */
	private final double[][] Q;
	/** V-Function represented as a vector */
	private final double[] V;
	/** Deterministic policy represented as a vector */
	private final int[] pol;
	/** Maximum number of iterations */
	private final int maxIter;
	/** Convergence threshold */
	private double epsilon;
	/** Indicates the number of iteration after which the value converged
	 * during the last run of VI, or -1 if it did not converge */
	private int converged;
	
	public ValueIteration(DiscreteMDP mdp, DiscountFactor gamma,
			int maxIter, double epsilon) {
		if(maxIter <= 0.) {
			throw new RuntimeException("The max number of iterations must be"
				+" greater than zero");
		}
		this.mdp = mdp;
		this.gamma = gamma;
		this.maxIter = maxIter;
		this.epsilon = epsilon;
		Q = new double[mdp.n][mdp.m];
		V = new double[mdp.n];
		pol = new int[mdp.n];
		// Perform Value Iteration
		performVI();
	}
	
	public final boolean performVI() {
		// (Re)initialize the state-action values to zero
		ArrUtils.zeros(Q);
		final double[][] Qold = ArrUtils.emptyLike(Q);
		
		// (Re)initialize the state values to zero
		ArrUtils.zeros(V);
		
		// Initialize the policy with a random action for all states
		for(int x=0; x<mdp.n; x++) {
			pol[x] = RandUtils.nextInt(mdp.m);
		}

		// Value iteration loop
		boolean quit = false;
		int iter = 0;
		while(!quit) {
		    iter++;
		    // Copy Q to Qold
		    ArrUtils.copyMatrix(Q, Qold, mdp.n, mdp.m);
		    // Compute the maximum of Q for each state, that is the optimal 
		    // state value V
		    for(int x=0; x<mdp.n; x++) {
		    	V[x] = ArrUtils.max(Q[x]);
		    }
		    // Perform VI update
		    mdp.computeQfromV(V,gamma,Q);
		    
		    // Compute the corresponding policy
		    mdp.computeGreedyPolicy(Q, pol);

		    if(epsilon >= 0) {
			    // If the update was small enough, we stop now
			    double delta = 0.;
			    for(int x=0; x<mdp.n; x++) {
			    	for(int u=0; u<mdp.m; u++) {
			    		delta += Math.pow(Q[x][u]-Qold[x][u],2);
			    	}
			    }
			    if(delta < epsilon) {
			    	converged = iter;
			       quit = true;
			    }
		    }
		    
		    if(iter == maxIter) {
		    	converged = -1;
		    	quit = true;
		    }
		}
		// Compute the state value of the final policy
	    for(int x=0; x<mdp.n; x++) {
	    	V[x] = Q[x][pol[x]];
	    }
	    return (converged >= 0);
	}
	
	public final double[][] getQ() {
		return Q;
	}
	
	public final double[] getV() {
		return V;
	}
	
	public int[] getPol() {
		return pol;
	}
	
	public final boolean hasConverged() {
		return (converged >= 0);
	}
	
	public final int convergedAfter() {
		return converged;
	}
}
