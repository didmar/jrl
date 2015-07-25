package com.github.didmar.jrl.mdp.dp;

import com.github.didmar.jrl.mdp.DiscreteMDP;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

import Jama.Matrix;

/**
 * Implements Policy Iteration.
 * @author Didier Marin
 */
public final class PolicyIteration {

	private final DiscreteMDP mdp;
	/** Discount factor */
	private final DiscountFactor gamma;
	/** Q-Function represented as a matrix */
	private final double[][] Q;
	/** V-Function represented as a vector */
	private double[] V;
	/** Deterministic policy represented as a vector */
	private final int[] pol;
	/** Maximum number of iterations */
	private final int maxIter;
	/** Indicates the number of iteration after which the policy converged
	 * during the last run of PI, or -1 if it did not converge */
	private int converged;

	public PolicyIteration(DiscreteMDP mdp, DiscountFactor gamma, int maxIter) {
		if(maxIter <= 0.) {
			throw new RuntimeException("The max number of iterations must be"
				+" greater than zero");
		}
		this.mdp = mdp;
		this.gamma = gamma;
		this.maxIter = maxIter;
		V = null;
		Q = new double[mdp.n][mdp.m];
		pol = new int[mdp.n];
		// Perform Policy Iteration
		performPI();
	}

	/**
	 * Performs Policy Iteration.
	 * @return true if the policy has converged, false else.
	 */
	public final boolean performPI() {
		// (Re)initialize the state-action values to zero
		ArrUtils.zeros(Q);

		// Initialize the policy with a random action for all states
		for(int x=0; x<mdp.n; x++) {
			pol[x] = RandUtils.nextInt(mdp.m);
		}
		final int[] polOld = pol.clone();

		final double[][] R = ArrUtils.zeros(mdp.n,1);

		final double[][] IminusGammaP = new double[mdp.n][mdp.n];

		// Value iteration loop
		boolean quit = false;
		boolean polUnchanged = true;
		int iter = 0;
		while(!quit) {
		    iter++;

		    // Evalute the current policy
		    for(int x=0; x<mdp.n; x++) {
		        R[x][0] = mdp.R[x][pol[x]];
		        for(int xn=0; xn<mdp.n; xn++) {
		            double xEqualsXn = 0.;
		            if(x==xn) {
		            	xEqualsXn = 1.;
		            }
		            IminusGammaP[x][xn] = xEqualsXn - gamma.value * mdp.P[x][pol[x]][xn];
		        }
		    }
		    // V = IminusGammaP^-1 R
		    Matrix invIminusGammaP = null;
			try {
				invIminusGammaP = ArrUtils.pinv(new Matrix(IminusGammaP));
			} catch (Exception e) {
				// FIXME handle this exception properly
				throw new RuntimeException("Could not compute pseuso-inverse of IminusGammaP");
			}
		    V = invIminusGammaP.times(new Matrix(R)).transpose().getArray()[0];

		    // Perform PI update
		    mdp.computeQfromV(V, gamma, Q);
		    // polOld <- pol
		    System.arraycopy(pol, 0, polOld, 0, mdp.n);
		    // Compute the policy according to the new Q
		    for(int x=0; x<mdp.n; x++) {
		    	pol[x] = ArrUtils.argmax(Q[x]);
		    }

		    // Check stopping conditions : the policy did not change or
		    // the maximum number of iterations is reached
		    polUnchanged = true;
		    for(int x=0; x<mdp.n; x++) {
		    	if(pol[x] != polOld[x]) {
		    		polUnchanged = false;
		    		break;
		    	}
		    }
		    if(polUnchanged || iter == maxIter) {
		        quit = true;
		    }
		}
		// Get the state value V of the final policy
	    for(int x=0; x<mdp.n; x++) {
	    	V[x] = Q[x][pol[x]];
	    }
	    if(iter == maxIter) {
	    	converged = -1;
	    } else {
	    	converged = iter;
	    }
	    return hasConverged();
	}

	public final double[][] getQ() {
		return Q;
	}

	public final double[] getV() {
		return V;
	}

	public final int[] getPol() {
		return pol;
	}

	public final boolean hasConverged() {
		return (converged >= 0);
	}

	public final int convergedAfter() {
		return converged;
	}
}

