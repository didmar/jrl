package com.github.didmar.jrl.policy;

/**
 * A policy is a distribution over actions given a state (and possibly a time
 * step). Note that we divide the job in two methods : 
 * {@link #computePolicyDistribution(double[])} should compute the action
 * distribution according to the state and time step, and
 * {@link #drawAction()} should draw an action according to that distribution.
 * This means that we can avoid recomputing the distribution if the state and
 * time step did not change, or if we want to draw more than one action.
 * 
 * @author Didier Marin
 */
public interface Policy {
	
	/**
     * Compute the policy distribution for a given state and time step
     * @param x   a state
     */
    public void computePolicyDistribution(final double[] x);
    
    /**
	 * @return an action drawn from the current policy distribution.
	 */
    public double[] drawAction();
    
}