package com.github.didmar.jrl.agent;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.policy.ParametricPolicy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.PGPEParametersDistribution;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Policy Gradient with Parameter-based exploration
 * 
 * @author Didier Marin
 */
public final class PGPE extends LearningAgent {

	/** Reward discount factor */
	private final DiscountFactor gamma;
	/** Number of sample episodes used for each update */
	private final int nEpiPerUpdate;
	/** The policy parameters distribution we will update */
	private final PGPEParametersDistribution paramsDist;
	/** The policy parameters we are trying at the current iteration */
	private final double[][] thetas;
	/** Performance for each policy we are testing (nEpiPerUpdate vector) */
	private final double[] R;
	/** Episode counter */
	private int e;
	/** Step counter */
	private int t;
	
	@SuppressWarnings("null")
	public PGPE(ParametricPolicy pol, DiscountFactor gamma, int nEpiPerUpdate,
			double[] sigma, double minSigma, StepSize stepSize) {
		super(pol);
		this.gamma = gamma;
		this.nEpiPerUpdate = nEpiPerUpdate;
		paramsDist = new PGPEParametersDistribution(pol.getParams(), sigma,
				minSigma, stepSize);
		// Generate some policy parameters using the distribution
		thetas = paramsDist.drawParameters(nEpiPerUpdate);
		// Bound these parameters  
		for(int i=0; i<nEpiPerUpdate; i++) {
			pol.boundParams(thetas[i]);
		}
		R = ArrUtils.zeros(nEpiPerUpdate);
		//meanR = new double[nEpiPerUpdate];
		e = 0; // set the episode counter to zero
		t = 0; // set the step counter to zero
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	@SuppressWarnings("null")
	public final void newEpisode(@NonNull final double[] x0, int maxT) {
		// Set the parameters we want to test for this episode  
        ((ParametricPolicy)pol).setParams(thetas[e]);
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	public final void receiveSample(@NonNull double[] x,
									@NonNull double[] u,
									@NonNull double[] xn,
									double r, boolean isTerminal) {
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
        if(e == nEpiPerUpdate) {
        	// Update the parameters distribution using an estimation of
        	// the performance gradient
        	paramsDist.updateParamsDistribution(R, thetas);
        	// Bound the mean of the updated distribution 
        	((ParametricPolicy)pol).boundParams(paramsDist.getMean());
        	// Draw new parameters for the next iteration,
        	// according to the updated distribution
        	paramsDist.drawParameters(thetas);
        	// Bound these parameters
    		for(int i=0; i<nEpiPerUpdate; i++) {
    			((ParametricPolicy)pol).boundParams(thetas[i]);
    		}
        	// Reset the episode counter
    		e = 0;
    		// Reset the array that stores the performances
    		ArrUtils.zeros(R);
        }
        
        // Use the mean of the parameters distribution
        ((ParametricPolicy)pol).setParams(paramsDist.getMean());
	}
	
	public final PGPEParametersDistribution getParamsDist() {
		return paramsDist;
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	@NonNull
	public final String toString() {
		return "Policy Gradient with Parameters Exploration";
	}
}
