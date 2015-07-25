package com.github.didmar.jrl.mdp.dp;

import com.github.didmar.jrl.mdp.DiscreteMDP;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

import Jama.Matrix;

// TODO check that it works
/**
 * Computes Q and V for a given policy (as a table).
 * @author Didier Marin
 */
public final class PolicyEvaluation {
	
	private final DiscreteMDP mdp;
	/** Discount factor */
	private final DiscountFactor gamma;
	/** The policy to evaluate */
	private final double[][] pol;
	/** Q-Function represented as a matrix */
	private final double[][] Q;
	/** V-Function represented as a vector */
	private double[] V;
	/** Transition kernel of the policy */
	private final double[][] K;

	public PolicyEvaluation(DiscreteMDP mdp, double[][] pol,
			DiscountFactor gamma) {
		this.mdp = mdp;
		this.pol = pol;
		this.gamma = gamma;
		Q = new double[mdp.n][mdp.m];
		K = new double[mdp.n][mdp.n];
		mdp.computeTransitionKernel(pol, K);
		// Perform Policy Iteration
		evaluate();
	}
	
	public final void evaluate() {
		// (Re)initialize the state-action values to zero
		ArrUtils.zeros(Q);
		
		final double[][] R = ArrUtils.zeros(mdp.n,1);
		final double[][] IminusGammaP = new double[mdp.n][mdp.n];
		
	    for(int x=0; x<mdp.n; x++) {
	        R[x][0] = ArrUtils.dotProduct(pol[x], mdp.R[x], mdp.m);
	        for(int xn=0; xn<mdp.n; xn++) {
	            double xEqualsXn = 0.;
	            if(x==xn) {
	            	xEqualsXn = 1.;
	            }
	            IminusGammaP[x][xn] = xEqualsXn - gamma.value * K[x][xn];
	        }
	    }
	    // V = IminusGammaP^-1 R
	    //Matrix invIminusGammaP = Utils.pinv(new Matrix(IminusGammaP));
	    
	    //double[][] invIminusGammaPArray = Utils.cloneMatrix(IminusGammaP);
	    //Utils.slowInverse(IminusGammaP, invIminusGammaPArray, mdp.n);
	    //Matrix invIminusGammaP = new Matrix(invIminusGammaPArray);
	    
	    Matrix invIminusGammaP = new Matrix(IminusGammaP).inverse();
	    
	    V = invIminusGammaP.times(new Matrix(R)).transpose().getArray()[0];
	    
	    mdp.computeQfromV(V, gamma, Q);
	}
	
	public final double[][] getQ() {
		return Q;
	}
	
	public final double[] getV() {
		return V;
	}
}
