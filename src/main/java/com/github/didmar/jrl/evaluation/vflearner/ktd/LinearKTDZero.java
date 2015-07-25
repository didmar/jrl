package com.github.didmar.jrl.evaluation.vflearner.ktd;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.VFunction;
import com.github.didmar.jrl.evaluation.vflearner.VFunctionLearner;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * An implementation of KTD-V(0) for linear state value function approximators,
 * as in Geist et al. "Differences Temporelles de Kalman", algorithm 2. Does not
 * work with stochastic transitions !
 * @author Didier Marin
 */
public final class LinearKTDZero implements VFunctionLearner, EnvironmentListener {

	/** Linear state value function approximation */
	private final LinearVFunction vFunction;
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
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Temporary n-length vector */
	private final double[] H;
	/** Used to store the state features of the next state */
	private final double[] phixn;
	/** Parameters prediction */
	private final double[] s;
	private final double[] P_s_r;
	private final double[] K;
	private final double[] K_P_r;
	private final double[][] K_P_r_K_T;
	
	/**
	 * Construct a {@link LinearKTDZero}.
	 * @param vFunction    linear state value function approximation
	 * @param gamma        reward discounted factor
	 * @param P_evo_init   diagonal value of the initial evolution noise
	 *                     variance matrix
	 * @param P_evo_step   step of the evolution noise variance matrix
	 * @param P_obs_step   step of the observation noise
	 * @throws Exception 
	 */
	public LinearKTDZero(LinearVFunction vFunction, DiscountFactor gamma,
			double P_evo_init, double P_evo_step, double P_obs_step) {
		this.vFunction = vFunction;
		this.gamma = gamma;
		n = vFunction.getParamsSize();
		P_evo = ArrUtils.eye(n,P_evo_init);
		this.P_evo_step = P_evo_step;
		this.P_obs_step = P_obs_step;
	    
	    // arrays for temporary storage to avoid mem. alloc.
		H = new double[n];
		phixn = new double[n];
		s = new double[n];
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
	    // Compute statistics : H = phi(x) - gamma * phi(xn)
	    vFunction.getFeatures().phi(x, H);
	    vFunction.getFeatures().phi(xn, phixn);
	    if(!isTerminal) {
	        for(int i=0; i<n; i++) {
	        	H[i] -= gamma.value * phixn[i];
	        }
	    }
	    final double r_predict = ArrUtils.dotProduct(H, s, n);
	    ArrUtils.multiply(P_evo, H, P_s_r, n, n);
        final double P_r = ArrUtils.dotProduct(H, P_s_r, n) + P_obs_step;
        // Compute optimal gain K and update the extended parameters accordingly
        double tdErr = r - r_predict;
        for(int i=0; i<n; i++) {
        	K[i] = P_s_r[i] / P_r;
        	s[i] += K[i] * tdErr;
        	K_P_r[i] = K[i] * P_r;
        }
        // P_evo <- P_evo - K_P_r . K^T
        ArrUtils.multiply(K_P_r, K, K_P_r_K_T, n);
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
		return "LinearKTD(0)";
	}
}
