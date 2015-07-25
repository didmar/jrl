package com.github.didmar.jrl.evaluation.vflearner.lstd;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.evaluation.valuefunction.VFunction;
import com.github.didmar.jrl.evaluation.vflearner.QFunctionLearner;
import com.github.didmar.jrl.evaluation.vflearner.VFunctionLearner;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * An incremental LSTD method that learns both the advantage and the state-value
 * function. See Peters and Schaal 2008 "Natural actor-critic".
 * @author Didier Marin
 */
public final class ILSTDAV implements QFunctionLearner, VFunctionLearner,
	EnvironmentListener {

	/** Advantage function approximation */
	private final LinearQFunction aFunction;
	/** Advantage function approximation */
	private final LinearVFunction vFunction;
	/** Number of advantage function approximation parameters */
	private final int m;
	/** Number of state value function approximation parameters */
	private final int n;
	/** Reward discount factor */
	private final DiscountFactor gamma;
	/** Eligibility factor */
	private final DiscountFactor lambda;
	/** Number of steps between each parameters update */
	private final int nbStepsBeforeUpdate;
	/** Number of steps before the next update */
	private int stepsBeforeUpdate;
	/** Initial value of the diagonal of Ainv */
	private double diagAinv0;
	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;
	
	private final double[][] Ainv;
	private final double[] b;
	private final double[] z;
	
	private final double[] xu;
	private final double[] phix;
	private final double[] phixn;
	private final double[] psixu;
	private final double[] psiphix;
	private final double[] psiphixMinusGammaPsiphixn;
	private final double[] tmp;
	
	public ILSTDAV(LinearQFunction aFunction, LinearVFunction vFunction,
			DiscountFactor gamma, DiscountFactor lambda, int nbStepsBeforeUpdate,
			double diagAinv0) {
		if(diagAinv0 <= 0.) {
			throw new IllegalArgumentException("The initial value of the diagonal of the"
					+" statistics matrix must be positive");
		}
		this.aFunction = aFunction;
		this.vFunction = vFunction;
		this.gamma = gamma;
		this.lambda = lambda;
		this.nbStepsBeforeUpdate = nbStepsBeforeUpdate;
		stepsBeforeUpdate = this.nbStepsBeforeUpdate;
		this.diagAinv0 = diagAinv0;
		m = aFunction.getParamsSize();
		n = vFunction.getParamsSize();
		Ainv = ArrUtils.eye(m+n, diagAinv0);
		b = ArrUtils.zeros(m+n);
		z = ArrUtils.zeros(m+n);
		xDim = aFunction.getXDim();
		uDim = aFunction.getUDim();
		
		xu   = new double[xDim+uDim];
		psixu    = new double[m];
		phix     = new double[n];
		phixn    = new double[n];
		psiphix  = new double[m+n];
		psiphixMinusGammaPsiphixn = new double[m+n];
		tmp = new double[m+n];
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	@Override
	public final void newEpisode(double[] x0, int maxT) {
		// Reinitialize eligibility traces vector
		ArrUtils.zeros(z);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	@Override
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Compute the features
        System.arraycopy(x,  0,   xu,    0, xDim);
        System.arraycopy(u,  0,   xu, xDim, uDim);
        aFunction.getFeatures().phi(  xu,   psixu);
        vFunction.getFeatures().phi(   x,    phix);
        System.arraycopy(psixu,  0,  psiphix, 0, m);
        System.arraycopy(phix,   0,  psiphix, m, n);
        
        // Update the statistics
        for(int i=0; i<m+n; i++) {
        	z[i] = lambda.value*gamma.value*z[i] + psiphix[i];
        }
        System.arraycopy(psiphix, 0, psiphixMinusGammaPsiphixn, 0, m+n);
        if(!isTerminal) {
        	vFunction.getFeatures().phi(xn, phixn);
        	for(int i=0; i<n; i++) {
        		psiphixMinusGammaPsiphixn[m+i] -= gamma.value * phixn[i];
        	}
        }
        try {
			ArrUtils.shermanMorrisonFormula(Ainv, z, psiphixMinusGammaPsiphixn, m+n);
		} catch (Exception e) {
			e.printStackTrace();
		}
        for(int i=0; i<m+n; i++) {
        	b[i] += z[i]*r;
        }
        if(nbStepsBeforeUpdate > 0) {
	        // Decrease the update counter, if it reaches zero compute the updated parameters
	        stepsBeforeUpdate--;
			if(stepsBeforeUpdate == 0) {
				computeValueParameters();
				stepsBeforeUpdate = nbStepsBeforeUpdate;
			}
        }

	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	@Override
	public final void endEpisode() {
		// Nothing to do
	}

	/**
	 * Update the parameters of both value functions approximations,
	 * based on the statistics.
	 */
	public void computeValueParameters() {
		ArrUtils.multiply(Ainv, b, tmp, m+n, m+n);
		// Extract the advantage function parameters and the state value
		// function parameters, using psixu and phixu for temporary storage
		System.arraycopy(tmp, 0, psixu, 0, m);
		System.arraycopy(tmp, m, phix,  0, n);
		aFunction.setParams(psixu);
		vFunction.setParams(phix);
	}
	
	/**
	 * Partially forget the statistics, with factor kappa in [0,1].
	 * For kappa=0, stats are completely forgotten.
	 * Useful when the policy changes by small increments.
	 * @param kappa forget factor
	 */
	public final void applyForgetFactor(DiscountFactor kappa) {
		for(int i=0; i<m+n; i++) {
			z[i] *= kappa.value;
			b[i] *= kappa.value;
			Ainv[i][i] = kappa.mixture(Ainv[i][i], diagAinv0);
		}
	}
	
	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.QFunctionLearner#getQFunction()
	 */
	@Override
	public final QFunction getQFunction() {
		return aFunction;
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.VFunctionLearner#getVFunction()
	 */
	@Override
	public final VFunction getVFunction() {
		return vFunction;
	}

}
