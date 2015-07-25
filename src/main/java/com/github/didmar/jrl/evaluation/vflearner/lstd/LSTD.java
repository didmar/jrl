package com.github.didmar.jrl.evaluation.vflearner.lstd;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.VFunction;
import com.github.didmar.jrl.evaluation.vflearner.VFunctionLearner;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

import Jama.Matrix;

// TODO add a reference to the algo
// TODO inherit from TD ?
/**
 * Least-Square Temporal Difference learner.
 * @author Didier Marin
 */
public final class LSTD implements VFunctionLearner, EnvironmentListener {

	/** State value function approximation */
	private final LinearVFunction vFunction;
	/** Number of state value function approximation parameters */
	private final int n;
	/** Reward discount factor */
	private final DiscountFactor gamma;
	/** Number of steps between each parameters update */
	private final int nbStepsBeforeUpdate;
	/** Number of steps before the next update */
	private int stepsBeforeUpdate;
	/** Regularization factor */
	private final double regFactor;
	
	private final double[][] A;
	
	private final double[] b;
	
	// used for temporary storage
	private final double[] phixMinusGammaPhixn;
	private final double[][] Aupdate;
	

	public LSTD(LinearVFunction vFunction, DiscountFactor gamma,
			int nbStepsBeforeUpdate, double regFactor) {
		this.vFunction = vFunction;
		this.gamma = gamma;
		this.nbStepsBeforeUpdate = nbStepsBeforeUpdate;
		stepsBeforeUpdate = this.nbStepsBeforeUpdate;
		this.regFactor = regFactor;
		n = vFunction.getParamsSize();
        A = ArrUtils.zeros(n, n);
        b = ArrUtils.zeros(n);
        
        phixMinusGammaPhixn = new double[n];
    	Aupdate = new double[n][n];
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.VFunctionLearner#getVFunction()
	 */
	public final VFunction getVFunction() {
		return vFunction;
	}

	public final void newEpisode(double[] x0, int maxT) {
		// Nothing to do
	}

	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Update the statistics
		final double[] phix  = vFunction.getFeatures().phi(x);
        System.arraycopy(phix, 0, phixMinusGammaPhixn, 0, n);
        if(!isTerminal) {
        	final double[] phixn = vFunction.getFeatures().phi(xn);
        	for(int i=0; i<n; i++) {
            	phixMinusGammaPhixn[i] -= gamma.value*phixn[i];
            }
        }
        ArrUtils.multiply(phix, phixMinusGammaPhixn, Aupdate, n);
        for(int i=0; i<n; i++) {
        	for(int j=0; j<n; j++) {
        		A[i][j] += Aupdate[i][j];
        	}
        }
        for(int i=0; i<n; i++) {
        	b[i] += phix[i] * r;
        }
        // Add a L2 regularization term if a regularization factor was given
        if(regFactor > 0) {
        	for(int i=0; i<n; i++) {
        		A[i][i] += regFactor;
        	}
        }
        // Increase the step counter
        stepsBeforeUpdate--;
        // Check if we must update now
        if(stepsBeforeUpdate == 0) {
        	Matrix matA = new Matrix(A,n,n);
        	Matrix matB = new Matrix(b,n);
        	// v = A^-1 * B
        	vFunction.setParams(matA.solve(matB).transpose().getArray()[0]);
            // Reset the statistics
            ArrUtils.zeros(A);
            ArrUtils.zeros(b);
            // Reset the number of steps before the next update
            stepsBeforeUpdate = nbStepsBeforeUpdate;
        }
	}
	
	public final void endEpisode() {
		// Nothing to do
	}
	
	@Override
	public final String toString() {
		return "LSTD";
	}
}
