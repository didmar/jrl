package com.github.didmar.jrl.evaluation.vflearner.ktd;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.ParametricVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.VFunction;
import com.github.didmar.jrl.evaluation.vflearner.VFunctionLearner;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * KTD-V using an unscented transform, as in Geist et al. "Differences
 * Temporelles de Kalman", algorithm 3. The state value approximator must be
 * parametric ({@link ParametricVFunction}). Does not work with stochastic
 * transitions !
 * @author Didier Marin
 */
public final class KTDZero implements VFunctionLearner, EnvironmentListener {

	/** State value function approximation */
	private final ParametricVFunction vFunction;
	/** Number of value function parameters */
	private final int n;
	/** Reward discount factor */
	private final DiscountFactor gamma;
	/** Evolution noise variance matrix */
	private final double[][] P_evo;
	/** Step of the evolution noise variance matrix */
	private double P_evo_step;
	/** Step of the observation noise */
	private double P_obs_step;
	/** Sigma-points scaling factor (for the unscented transform) */
	private final double k;

	// arrays for temporary storage to avoid mem. alloc.
	/** Parameters prediction */
	private final double[] s;
	/** Temporary n-by-n matrix for computing Cholesky decomposition */
	private final double[][] L;
	/** Used to store the sigma-points (2*n-by-n matrix) */
	private final double[][] sigpts;
	/** Used to the weights of the sigma-points (2*n-length vector) */
	private final double[] w;
	/** Temporary n-by-n matrix */
	private final double[][] C;
	/** Temporary 2*n-length vector */
	private final double[] r_predict_sigpts;
	/** Temporary n-length vector */
	private final double[] vParams;

	private final double[] P_s_r;
	private final double[] K;
	private final double[] K_P_r;
	private final double[][] K_P_r_K_T;

	/**
	 * Construct a {@link KTDZero}.
	 * @param vFunction    state value function approximation
	 * @param gamma        reward discounted factor
	 * @param P_evo_init   diagonal value of the initial evolution noise
	 *                     variance matrix
	 * @param P_evo_step   step of the evolution noise variance matrix
	 * @param P_obs_step   step of the observation noise
	 * @param k            sigma-points scaling factor
	 * @throws Exception
	 */
	public KTDZero(ParametricVFunction vFunction, DiscountFactor gamma,
			double P_evo_init, double P_evo_step, double P_obs_step, double k) {
		this.vFunction = vFunction;
		this.gamma = gamma;
		n = vFunction.getParamsSize();
		P_evo = ArrUtils.eye(n, P_evo_init);
		this.P_evo_step = P_evo_step;
		this.P_obs_step = P_obs_step;
	    this.k = k;

	    // arrays for temporary storage to avoid mem. alloc.
	    s = new double[n];
	    L = new double[n][n];
	    sigpts = new double[2*n][n];
	    w = new double[2*n];
	    C = new double[n][n];
	    r_predict_sigpts = new double[2*n];
	    vParams = new double[n];
	    P_s_r = new double[n];
    	K = new double[n];
    	K_P_r = new double[n];
    	K_P_r_K_T = new double[n][n];
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.VFunctionLearner#getVFunction()
	 */
	public final VFunction getVFunction() {
		return vFunction;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(double[] x0, int maxT) {
		// Nothing to do
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	public final void receiveSample(double[] x, double[] u, double[] xn, double r, boolean isTerminal) {
		// Prediction step
	    System.arraycopy(vFunction.getParams(), 0, s, 0, n);
	    // Compute the prediction of the covariance of the evolution noise
	    for(int i=0; i<n; i++) {
	    	P_evo[i][i] += P_evo_step;
	    }
	    assert ArrUtils.isSymmetric(P_evo);
	    // Compute sigma-points (unscented transform)
	    // Reinitialize to zeros
        ArrUtils.zeros(sigpts);
        // Copy the mean of the parameters prediction
        System.arraycopy(s, 0, sigpts[0], 0, n);
        // Reinitialize the weights of the sigma-points to zeros
        ArrUtils.zeros(w);
        w[0] = k/(n+k);

	    for(int i=0; i<n; i++) {
        	for(int j=0; j<n; j++) {
        		C[i][j] = (n+k)*P_evo[i][j];
        	}
        }
	    // FIXME debug squareRootInPlace, does not work !
	    try {
			ArrUtils.squareRootInPlace(C,n);
		} catch (Exception e) {
			// FIXME handle this exception properly
			new RuntimeException("Could not compute square root of C");
		}

	    ArrUtils.choleskyDecomposition(C, L, n);

	    // For each sigma-points
        for(int i=0; i<n; i++) {
        	for(int j=0; j<n; j++) {
        		sigpts[i  ][j] = s[j] + L[j][i];
        		sigpts[i+n][j] = s[j] - L[j][i];
        	}
            w[i  ] = 1./(2.*(n+k));
            w[i+n] = 1./(2.*(n+k));
        }

        ArrUtils.zeros(r_predict_sigpts);
        for(int i=0; i<2*n; i++) {
        	System.arraycopy(sigpts[i], 0, vParams, 0, n);
        	vFunction.setParams(vParams);
        	final double Vx  = vFunction.get(x);
        	final double Vxn = vFunction.get(xn);
        	r_predict_sigpts[i] = Vx;
        	if(!isTerminal) {
        		r_predict_sigpts[i] -= gamma.value*Vxn;
        	}
        }

        // Compute statistics
        final double r_predict = ArrUtils.dotProduct(w, r_predict_sigpts, 2*n);
        ArrUtils.zeros(P_s_r);
        for(int i=0; i<2*n; i++) {
        	for(int j=0; j<n; j++) {
        		P_s_r[j] += w[i]*(sigpts[i][j]-s[j])*(r_predict_sigpts[i]-r_predict);
        	}
        }
        double P_r = P_obs_step;
        for(int i=0; i<2*n; i++) {
            P_r += w[i] * Math.pow(r_predict_sigpts[i]-r_predict, 2);
        }
        // Compute optimal gain K and update the extended parameters accordingly
        double tdErr = r - r_predict;
        for(int i=0; i<n; i++) {
        	K[i] = P_s_r[i] / P_r;
        	s[i] += K[i] * tdErr;
        	K_P_r[i] = K[i] * P_r;
        }
        // P_evo <- P_evo - K_P_r * K^T
        ArrUtils.multiply(K_P_r, K, K_P_r_K_T, n);
        assert ArrUtils.isSymmetric(K_P_r_K_T);
        for(int i=0; i<n; i++) {
        	for(int j=0; j<n; j++) {
        		P_evo[i][j] -= K_P_r_K_T[i][j];
        	}
        }
        // Update the advantage and state value approximation parameters
        vFunction.setParams(s);

	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		// Nothing to do
	}

	@Override
	public final String toString() {
		return "KTD(0)";
	}
}
