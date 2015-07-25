package com.github.didmar.jrl.agent;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.policy.ParametricPolicy;
import com.github.didmar.jrl.utils.CEParametersDistribution;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Cross Entropy Policy Search is similar to the Finite Difference method, 
 * but instead of weighting the policies by their performance it choose a
 * fraction of the best policies and update the prior to match their
 * distribution.
 * 
 * @author Didier Marin
 */
public final class CEPS extends LearningAgent {

	/** How many policies we draw from the prior and evaluate */
	private final int nPolEvalPerUpdate;
	/** How many episodes are used to evaluate one policy */
	private final int nEpiPerPolEval;
	/** How many of the best policies we should use to compute the new prior */
	private final int nSelectedPol;
	/** Discount factor */
	private final DiscountFactor gamma;
	/** Supplementary term added to sigma in order to avoid premature
	 * convergence */
	private final double noise;
	/** Indicates if the selection method is greedy, which means that we use
	 * the best parameters as the new mean and the additional noise as the new
	 * sigma of the CE distribution. */
	private final boolean greedy;
	/** Indicates if we should replace the first sample of the new distribution
	 * by the best parameters so far */
	private final boolean reuseTheBest;
	/** The policy parameters we are trying at the current iteration */
	private final double[][] thetas;
	/** The best policy parameters seen so far */
	private final double[] bestTheta;
	/** Length of the policy parameters */
	private final int n;
	/** The policy parameters distribution we will update */
	private final CEParametersDistribution paramsDist;
	/** Weighting of each learning episode performance
	 * (nPolEvalPerUpdate*nEpiPerPolEval vector) */
	private final double[] weightR;
	/** Performance of each learning episode
	 * (nPolEvalPerUpdate*nEpiPerPolEval vector) */
	private final double[] R;
	/** Mean performance for each policy we're testing
	 * (nPolEvalPerUpdate vector) */
	private final double[] meanR;
	/** Episode counter */
	private int e;
	/** Step counter */
	private int t;
	
	@SuppressWarnings("null")
	public CEPS(ParametricPolicy pol, int nPolEvalPerUpdate, int nEpiPerPolEval,
				int nSelectedPol, DiscountFactor gamma, double[] sigma, double noise,
				boolean greedy, boolean reuseTheBest) {
		super(pol);
		this.nPolEvalPerUpdate = nPolEvalPerUpdate;
		this.nEpiPerPolEval = nEpiPerPolEval;
		this.nSelectedPol = nSelectedPol;
		this.noise = noise;
		this.greedy = greedy;
		this.reuseTheBest = reuseTheBest;
		this.gamma = gamma;
		final double[] thetaInit = pol.getParams();
		n = thetaInit.length;
		paramsDist = new CEParametersDistribution(thetaInit, sigma);
		// Generate some policy parameters using the initial distribution
		thetas = paramsDist.drawParameters(nPolEvalPerUpdate);
		// Bound these parameters
		for(int i=0; i<nPolEvalPerUpdate; i++) {
			pol.boundParams(thetas[i]);
		}
		bestTheta = ArrUtils.cloneVec(thetaInit);
		// Reuse the best params ?
    	if(reuseTheBest) {
    		System.arraycopy(bestTheta, 0, thetas[0], 0, bestTheta.length);
    	}
		
    	weightR = ArrUtils.ones(nEpiPerPolEval);
		R = ArrUtils.zeros(nPolEvalPerUpdate*nEpiPerPolEval);
		meanR = new double[nPolEvalPerUpdate];
		e = 0;
		t = 0;
	}

	/**
	 * Variant where the initial samples are drawn from a uniform distribution
	 * within the policy parameters bounds.
	 */
	@SuppressWarnings("null")
	public CEPS(ParametricPolicy pol, int nPolEvalPerUpdate, int nEpiPerPolEval,
			int nSelectedPol, DiscountFactor gamma, double[][] thetaBounds,
			double noise, boolean greedy, boolean reuseTheBest) {
		super(pol);
		this.nPolEvalPerUpdate = nPolEvalPerUpdate;
		this.nEpiPerPolEval = nEpiPerPolEval;
		this.nSelectedPol = nSelectedPol;
		this.noise = noise;
		this.greedy = greedy;
		this.reuseTheBest = reuseTheBest;
		this.gamma = gamma;
		final double[] thetaInit = pol.getParams();
		n = thetaInit.length;
		paramsDist = new CEParametersDistribution(thetaInit, ArrUtils.ones(thetaInit.length));
		// Generate some policy parameters uniformly within the given bounds
		thetas = new double[nPolEvalPerUpdate][thetaInit.length];
		for(int i=0; i<nPolEvalPerUpdate; i++) {
			for(int j=0; j<thetaInit.length; j++) {
				thetas[i][j] = thetaBounds[j][0]
				     + RandUtils.nextDouble()*(thetaBounds[j][1]-thetaBounds[j][0]); 
			}
		}
		// Bound these parameters  
		for(int i=0; i<nPolEvalPerUpdate; i++) {
			pol.boundParams(thetas[i]);
		}
		bestTheta = ArrUtils.cloneVec(thetaInit);
		
		weightR = ArrUtils.ones(nEpiPerPolEval);
		R = ArrUtils.zeros(nPolEvalPerUpdate*nEpiPerPolEval);
		meanR = new double[nPolEvalPerUpdate];
		e = 0;
		t = 0;
	}
	
	/**
	 * Change the std dev of the parameters distribution and redraw new policy
	 * parameters according to the new distribution.
	 * @param sigma	the new std dev of the parameters distribution 
	 */
	public final void setSigma(double[] sigma) {
		paramsDist.setSigma(sigma);
		// Redraw the policy parameters according to the new distribution
    	paramsDist.drawParameters(thetas);
	}
	
	/**
	 * Change the mean of the parameters distribution and redraw new policy
	 * parameters according to the new distribution.
	 * @param sigma	the new std dev of the parameters distribution 
	 */
	public final void setMean(double[] mean) {
		paramsDist.setMean(mean);
		// Redraw the policy parameters according to the new distribution
    	paramsDist.drawParameters(thetas);
	}
	
	/**
	 * Returns the std dev of the parameters distribution.
	 * @return the std dev of the parameters distribution
	 */
	public final double[] getSigma() {
		return paramsDist.getSigma();
	}
	
	/**
	 * Returns the mean of the parameters distribution.
	 * @return the mean of the parameters distribution
	 */
	public final double[] getMean() {
		return paramsDist.getMean();
	}
	
	public final void setPerfWeight(double[] w) {
		assert w.length == weightR.length;
		
		System.arraycopy(w, 0, weightR, 0, weightR.length);
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	@SuppressWarnings("null")
	public final void newEpisode(@NonNull double[] x0, int maxT) {
		// Get the index of the parameters for this iteration
        final int k = e/nEpiPerPolEval;
        // Use these for the starting episode  
        ((ParametricPolicy)pol).setParams(thetas[k]);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double, double[])
	 */
	public final void receiveSample(@NonNull double[] x,
									@NonNull double[] u,
									@NonNull double[] xn,
									double r,
									boolean isTerminal) {
		// Add this reward to the discounted reward of the episode
		R[e] += Math.pow(gamma.value, t) * r;
		t++;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	@SuppressWarnings("null")
	public final void endEpisode() {
		e++;
		
		// Reset the step counter to zero 
		t = 0;
		
        // Test if we reached the number of episode needed for an update
        if(e == nPolEvalPerUpdate*nEpiPerPolEval) {
        	// Update the parameters distribution using Cross-Entropy,
        	// using the discounted rewards as the selection criterion
        	for(int i=0; i<nPolEvalPerUpdate; i++) {
        		double s = 0.;
        		for(int j=0; j<nEpiPerPolEval; j++) {
        			s += R[i*nEpiPerPolEval + j] * weightR[j];
        		}
        		s /= nEpiPerPolEval;
        		meanR[i] = s;
        	}
        	// Get the index of the best sample parameters
        	final int indBest = paramsDist.computeParamsDistribution(meanR,
        			thetas,	nSelectedPol, noise, greedy);
        	// Copy the best sample parameters
        	System.arraycopy(thetas[indBest], 0, bestTheta, 0, n);
        	// Bound the mean of the updated distribution 
        	((ParametricPolicy)pol).boundParams(paramsDist.getMean());
        	// Draw new parameters for the next iteration,
        	// according to the updated distribution
        	paramsDist.drawParameters(thetas);
        	// Bound these parameters  
    		for(int i=0; i<nPolEvalPerUpdate; i++) {
    			((ParametricPolicy)pol).boundParams(thetas[i]);
    		}
        	// Reuse the best params ?
        	if(reuseTheBest) {
        		System.arraycopy(bestTheta, 0, thetas[0], 0, bestTheta.length);
        	}
    		e = 0;
    		ArrUtils.zeros(R);
        }
        
        // Use the mean of the policy parameters distribution
        ((ParametricPolicy)pol).setParams(paramsDist.getMean());
	}

	public final CEParametersDistribution getParamsDist() {
		return paramsDist;
	}

	public final double[][] getThetas() {
		return thetas;
	}
	
	public final int getEpisodeCount() {
		return e;
	}
	
	public final int getStepCount() {
		return t;
	}
	
	/**
	 * Returns the mean performance of each candidate policy for the last iteration.
	 * @return the mean performance of each candidate policy for the last iteration
	 */
	public final double[] getMeanR() {
		return meanR;
	}
	
	@Override
	@NonNull
	public final String toString() {
		return "CEPS";
	}

}
