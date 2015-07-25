package com.github.didmar.jrl.evaluation.vflearner.ktd;

import com.github.didmar.jrl.environment.EnvironmentListener;
import com.github.didmar.jrl.evaluation.valuefunction.ParametricQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.ParametricVFunction;
import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.evaluation.valuefunction.VFunction;
import com.github.didmar.jrl.evaluation.vflearner.QFunctionLearner;
import com.github.didmar.jrl.evaluation.vflearner.VFunctionLearner;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Kalman Temporal Difference method for approximating both the advantage and
 * the state value function. See Geist 2009 PhD thesis "Optimisation des chaînes
 * de production dans l'industrie sidérurgique : une approche statistique de
 * l'apprentissage par renforcement" for the details.
 * @author Didier Marin
 */
public final class KTDAV implements QFunctionLearner, VFunctionLearner,
		EnvironmentListener {

	/** Advantage function approximation */
	private final ParametricQFunction aFunction;
	/** State value function approximation */
	private final ParametricVFunction vFunction;
	/** Number of state value function parameters */
	private final int n;
	/** Number of advantage approximation parameters */
	private final int m;
	/** Reward discount factor */
	private final DiscountFactor gamma;
	/** Scaling factor for the adaptative evolution noise variance matrix */
	private final double eta;
	/** Observation noise added at each step */
	private final double P_obs_step;
	/** Evolution noise variance matrix */
	private final double[][] P_evo;
	/** Step of the evolution noise variance matrix */
	private double[][] P_evo_step;
	/** Sigma-points scaling factor (for the unscented transform) */
	private final double k;
	/** Extended parameters prediction, composed of :
	 * - advantage function parameters (size m)
	 * - state value function approximation parameters (size n)
	 * - observation noise b (scalar)
	 * - observation noise n (scalar)
	 */
	private final double[] s;
	/** Evolution matrix */
	private final double[][] F;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Temporary m+n+2-length vector */
	private final double[] tmp;
	/** Temporary m+n+2-by-m+n+2 matrix for transposing F */
	private final double[][] F_T;
	/** Temporary m+n+2-by-m+n+2 matrix for computing P_evo . F_T */
	private final double[][] PF;
	/** Used to store the sigma-points (2*(m+n+2)-by-m+n+2 matrix) */
	private final double[][] sigpts;
	/** Used to the weights of the sigma-points (2*(m+n+2)-length vector) */
	private final double[] w;
	/** Temporary m+n+2-by-m+n+2 matrix for computing (m+n+k)*P_evo */
	private final double[][] mnk_P_evo;
	/** Temporary m+n+2-by-m+n+2 matrix for computing the Cholesky decomposition
	 * of mnk_P_evo */
	private final double[][] L;
	/** Temporary 2*(m+n+2)-length vector */
	private final double[] r_predict_sigpts;
	private final double[] aParams;
	private final double[] vParams;
	private final double[] P_s_r;
	private final double[] K;
	private final double[] K_P_r;
	private final double[][] K_P_r_K_T;

	/**
	 * Construct a {@link KTDAV}.
	 * @param aFunction        advantage function approximation
	 * @param vFunction        state value function approximation
	 * @param gamma            reward discount factor
	 * @param lambda           eligibility factor
	 * @param P_evo_init       initial evolution noise
	 * @param eta              scaling factor for the adaptative evolution noise
	 *                         variance matrix
	 * @param P_obs_step       observation noise added to P at each step
	 * @param k                sigma-points scaling factor (for the unscented
	 *                         transform)
	 * @param sigma_squared    prior on the variance of the residuals (should be
	 *                         ~ 1e-2)
	 * @throws Exception 
	 */
	public KTDAV(ParametricQFunction aFunction, ParametricVFunction vFunction,
			DiscountFactor gamma, DiscountFactor lambda, double P_evo_init,
			double eta,	double P_obs_step, double k, double sigma_squared) {
		this.aFunction = aFunction;
		this.vFunction = vFunction;
		this.gamma = gamma;
        this.eta = eta;
        this.P_obs_step = P_obs_step;
        this.k = k;
        
        n = this.vFunction.getParamsSize();
        m = this.aFunction.getParamsSize();
        
        P_evo = ArrUtils.zeros(m+n+2,m+n+2);
        for(int i=0; i<m+n+2; i++) {
        	P_evo[i][i] = P_evo_init;
        }
        
        P_evo_step = ArrUtils.zeros(m+n+2,m+n+2);
        // If non-adaptative noise
        if(this.eta == 0) {
            this.P_evo_step[m+n  ][m+n  ] = sigma_squared;
            this.P_evo_step[m+n+1][m+n  ] = sigma_squared;
            this.P_evo_step[m+n  ][m+n+1] = sigma_squared;
            this.P_evo_step[m+n+1][m+n+1] = sigma_squared;
        } else {
        	this.P_evo_step[m+n  ][m+n  ] = sigma_squared;
        	this.P_evo_step[m+n+1][m+n  ] = -gamma.value*sigma_squared;
        	this.P_evo_step[m+n  ][m+n+1] = -gamma.value*sigma_squared;
        	this.P_evo_step[m+n+1][m+n+1] = Math.pow(gamma.value,2)*sigma_squared;
        }
        s = ArrUtils.zeros(m+n+2);
        // Copy the advantage function parameters 
        System.arraycopy(aFunction.getParams(), 0, s, 0, m);
        // Copy the state value function parameters 
        System.arraycopy(vFunction.getParams(), 0, s, m, n);
        
        F = ArrUtils.eye(m+n+2);
        F[m+n  ][m+n  ] = gamma.value * lambda.value;
        F[m+n+1][m+n  ] = 0.;
        F[m+n  ][m+n+1] = -gamma.value * (1-lambda.value);
        F[m+n+1][m+n+1] = 0.;
        
        // arrays for temporary storage to avoid mem. alloc.
        tmp = new double[m+n+2];
        F_T = new double[m+n+2][m+n+2];
        PF  = new double[m+n+2][m+n+2];
        sigpts = new double[2*(m+n+2)][m+n+2];
        w = new double[2*(m+n+2)];
        mnk_P_evo = new double[m+n+2][m+n+2];
        L = new double[m+n+2][m+n+2];
        r_predict_sigpts = new double[2*(m+n+2)];
        aParams = new double[m];
        vParams = new double[n];
    	P_s_r = new double[m+n+2];
    	K = new double[m+n+2];
    	K_P_r = new double[m+n+2];
    	K_P_r_K_T = new double[m+n+2][m+n+2];
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.vflearner.QFunctionLearner#getQFunction()
	 */
	public final QFunction getQFunction() {
		return aFunction;
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
		// s <- F.s
        ArrUtils.multiplyQuad(F, s, tmp, m+n+2);
        System.arraycopy(tmp, 0, s, 0, m+n+2);
        for(int i=0; i<m+n; i++) {
        	for(int j=0; j<m+n; j++) {
        		P_evo_step[i][j] *= eta; 
        	}
        }
        ArrUtils.copyMatrix(F, F_T, m+n+2);
        ArrUtils.transposeQuadInPlace(F, m+n+2);
        ArrUtils.multiplyQuad(P_evo, F_T, PF, m+n+2);
        // Predict the covariance of the evolution noise
        // P_evo = (F.PF) + P_evo_step
        ArrUtils.multiplyQuad(F, PF, P_evo, m+n+2);
        for(int i=0; i<m+n+2; i++) {
        	for(int j=0; j<m+n+2; j++) {
        		P_evo[i][j] += P_evo_step[i][j];
        	}
        }
        // Compute sigma-points (unscented transform)
        // Reinitialize to zeros
        ArrUtils.zeros(sigpts);
        // Copy the mean of the parameters prediction
        System.arraycopy(s, 0, sigpts[0], 0, m+n+2);
        // Reinitialize the weights of the sigma-points to zeros
        ArrUtils.zeros(w);
        w[0] = k/(m+n+k);
        
        for(int i=0; i<m+n+2; i++) {
        	for(int j=0; j<m+n+2; j++) {
        		mnk_P_evo[i][j] = (m+n+k)*P_evo[i][j];
        	}
        }
        ArrUtils.choleskyDecomposition(mnk_P_evo, L, m+n+2);
        // For each sigma-points
        for(int i=0; i<m+n+2; i++) {
        	for(int j=0; j<m+n+2; j++) {
        		sigpts[i        ][j] = s[j] + L[j][i];
        		sigpts[i+(m+n+2)][j] = s[j] - L[j][i];
        	}
            w[i] = 1./(2.*(m+n+k));
            w[i+(m+n+2)] = 1./(2.*(m+n+k));
        }
        ArrUtils.zeros(r_predict_sigpts);
        for(int i=0; i<2*(m+n+2); i++) {
        	System.arraycopy(sigpts[i], 0, aParams, 0, m);
        	System.arraycopy(sigpts[i], m, vParams, 0, n);
        	aFunction.setParams(aParams);
        	vFunction.setParams(vParams);
        	final double Axu = aFunction.get(x, u);
        	final double Vx  = vFunction.get(x);
        	final double Vxn = vFunction.get(xn);
        	r_predict_sigpts[i] = Axu + Vx + sigpts[i][m+n+1];
        	if(!isTerminal) {
        		r_predict_sigpts[i] -= gamma.value*Vxn;
        	}
            /*r_predict_sigpts[i] = inner( aFunction.psi(x,u), sigpts[i,0:m] ) \
                + inner( (self.vFunction.phi(x) - gamma * vFunction.phi(xn)), \
                         sigpts[i,m:m+n] ) \
                + sigpts[i,m+n+1]*/
        }
        // Compute statistics
        final double r_predict = ArrUtils.dotProduct(w, r_predict_sigpts, m+n+2);
        ArrUtils.zeros(P_s_r);
        for(int i=0; i<2*(m+n+2); i++) {
        	for(int j=0; j<m+n+2; j++) {
        		P_s_r[j] += w[i]*(sigpts[i][j]-s[j])*(r_predict_sigpts[i]-r_predict);
        	}
        }
        double P_r = P_obs_step;
        for(int i=0; i<2*(m+n+2); i++) {
            P_r += w[i] * Math.pow(r_predict_sigpts[i]-r_predict, 2);
        }
        // Compute optimal gain K and update the extended parameters accordingly
        double tdErr = r - r_predict;
        for(int i=0; i<m+n+2; i++) {
        	K[i] = P_s_r[i] / P_r;
        	s[i] += K[i] * tdErr;
        	K_P_r[i] = K[i] * P_r;
        }
        // P_evo <- P_evo - K_P_r . K^T
        ArrUtils.multiply(K_P_r, K, K_P_r_K_T, m+n+2);
        for(int i=0; i<m+n+2; i++) {
        	for(int j=0; j<m+n+2; j++) {
        		P_evo[i][j] -= K_P_r_K_T[i][j];
        	}
        }
        // Update the advantage and state value approximation parameters
        System.arraycopy(s, 0, aParams, 0, m);
    	System.arraycopy(s, m, vParams, 0, n);
    	aFunction.setParams(aParams);
    	vFunction.setParams(vParams);
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		// Nothing to do
	}
}
