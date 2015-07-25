package com.github.didmar.jrl.evaluation;

import java.util.List;

import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Use the Episodic REINFORCE method to compute an estimation of the performance
 * gradient of a policy based on sample episodes following that policy. The
 * performance criterion is the discounted reward sum.
 * 
 * @author Didier Marin
 *
 */
public final class REINFORCEGradientEstimator {
	
	/** The policy followed during the sample episodes */
	private final ILogDifferentiablePolicy pol;
	private final DiscountFactor gamma;
	private final int maxNEpiPerUpdate;
	private final int n;
	
	private double[][] grad;
	private double[] R;
	private double[] b;
	private double[] dJ;
	
	public REINFORCEGradientEstimator(ILogDifferentiablePolicy pol,
			DiscountFactor gamma, int maxNEpiPerUpdate) {
		this.pol = pol;
		this.gamma = gamma;
		this.maxNEpiPerUpdate = maxNEpiPerUpdate;
		n = pol.getParamsSize();
		grad = new double[maxNEpiPerUpdate][n];
        R = new double[maxNEpiPerUpdate];
        b = new double[n];
        dJ = new double[n];
	}
	
	/**
	 * Returns an estimation of the performance gradient based on a list of
	 * sample episodes, using the Episodic REINFORCE method
	 * @param episodes   list of sample episodes
	 * @return performance gradient estimation
	 * @throws Exception
	 */
	public final double[] computeGradientEstimation(List<Episode> episodes)
			throws Exception {
		final int nEpi = episodes.size();
        if(nEpi > maxNEpiPerUpdate) {
        	throw new Exception("Too many sample episodes !");
        }
        // Init the summed policy gradient matrix to zeros
    	ArrUtils.zeros(grad);
    	// Init the performance vector to zeros
    	ArrUtils.zeros(R);
        // For each episode from the rollouts
        for(int i=0; i<nEpi; i++) {
        	// Get the i-th episode from the episodes list, its duration and
        	// its samples
        	final Episode epi = episodes.get(i);
        	final int T = epi.getT();
            final double[][] x = epi.getX();
            final double[][] u = epi.getU();
            final double[]   r = epi.getR();
            // For each decision step of the i-th episode
            for(int t=0; t<T; t++) {
            	@SuppressWarnings("null")
				final double[] dLogdTheta = pol.dLogdTheta(x[t],u[t]);
            	for(int j=0; j<n; j++) {
            		grad[i][j] += dLogdTheta[j];
            	}
                // Update the sum of discounted rewards
                R[i] += Math.pow(gamma.value,t) * r[t];
            }
        }
        
        // Compute the optimal baseline
        ArrUtils.zeros(b);
		for(int j=0; j<n; j++) {
			double bNum   = 0.;
			double bDenum = 0.;
			for(int i=0; i<nEpi; i++) {
				bNum += Math.pow(grad[i][j],2)*R[i];
				bDenum += Math.pow(grad[i][j],2);
			}
			if(bDenum != 0.) {
				b[j] = bNum / bDenum;
			}
		}
        
        // Compute the performance gradient
        ArrUtils.zeros(dJ);
        for(int j=0; j<n; j++) {
        	for(int i=0; i<nEpi; i++) {
        		dJ[j] += grad[i][j]*(R[i]-b[j]);
        	}
        	dJ[j] /= nEpi;
        }
        
        return dJ;
	}
	
}
