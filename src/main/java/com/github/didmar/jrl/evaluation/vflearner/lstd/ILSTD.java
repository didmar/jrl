package com.github.didmar.jrl.evaluation.vflearner.lstd;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.VFunction;
import com.github.didmar.jrl.evaluation.vflearner.VFunctionLearner;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Incremental Least-Square Temporal Difference is an incremental variant of
 * LSTD where the state value approximation parameters can be computed at any
 * time. We use Sherman-Morrison lemma to avoid a matrix pseudo-inversion
 * every time we want to compute these parameters.
 * @author Didier Marin
 */
public final class ILSTD implements VFunctionLearner, EnvironmentListener {

	/** State value function approximation */
	private final LinearVFunction vFunction;
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
	
	private final double[][] Ainv;
	private final double[] b;
	private final double[] z;
	
	private final double[] phixMinusGammaPhixn;
	private final double[] tmp;

	public ILSTD(LinearVFunction vFunction, DiscountFactor gamma,
			DiscountFactor lambda, int nbStepsBeforeUpdate, double diagAinv0) {
		if(diagAinv0 <= 0.) {
			throw new IllegalArgumentException("The initial value of the diagonal of the"
					+" statistics matrix must be positive");
		}
		this.vFunction = vFunction;
		this.gamma = gamma;
		this.lambda = lambda;
		this.nbStepsBeforeUpdate = nbStepsBeforeUpdate;
		stepsBeforeUpdate = this.nbStepsBeforeUpdate;
		n = vFunction.getParamsSize();
		// Initialize the inverted statistics matrix A
		Ainv = ArrUtils.eye(n,diagAinv0);
		b = ArrUtils.zeros(n);
		z = ArrUtils.zeros(n);
		
		phixMinusGammaPhixn = new double[n];
		tmp = new double[n];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(double[] x0, int maxT) {
		// Reinitialize eligibility traces vector
		ArrUtils.zeros(z);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	public final void receiveSample(double[] x, double[] u, double[] xn,
			double r, boolean isTerminal) {
		// Update the statistics z, Ainv and b
		final double[] phix = vFunction.getFeatures().phi(x);
		for(int i=0; i<n; i++) {
        	z[i] = lambda.value*gamma.value*z[i] + phix[i];
		}
		System.arraycopy(phix, 0, phixMinusGammaPhixn, 0, n);
		if(!isTerminal) {
			final double[] phixn = vFunction.getFeatures().phi(xn);
			for(int i=0; i<n; i++) {
				phixMinusGammaPhixn[i] -= gamma.value*phixn[i];
			}
        }
		try {
			ArrUtils.shermanMorrisonFormula(Ainv, z, phixMinusGammaPhixn, n);
		} catch (Exception e) {
			e.printStackTrace();
		}
        for(int i=0; i<n; i++) {
        	b[i] += z[i] * r;
        }
        // Decrease the update counter, if it reaches zero compute the updated parameters
		stepsBeforeUpdate--;
		if(stepsBeforeUpdate == 0) {
			computeValueParameters();
			stepsBeforeUpdate = nbStepsBeforeUpdate;
		}
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		// Nothing to do
	}
	
	/**
	 * Update the value function approximation parameters based on the
	 * statistics.
	 */
	public final void computeValueParameters() {
		ArrUtils.multiply(Ainv, b, tmp, n, n);
		vFunction.setParams(tmp);
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.VFunctionLearner#getVFunction()
	 */
	public final VFunction getVFunction() {
		return vFunction;
	}
	
	@Override
	public final String toString() {
		return "ILSTD("+lambda.value+")";
	}
}
