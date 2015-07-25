package com.github.didmar.jrl.policy;

import org.eclipse.jdt.annotation.NonNull;

/**
 * A partial implementation of the ILogDifferentiable, with an explicit
 * representation of the policy parameters vector. The parameters bounding
 * method is left empty and might overridden. The only unimplemented method
 * is the one that returns the derivative of the log policy.
 * 
 * @author Didier Marin
 */
public abstract class LogDifferentiablePolicy implements ILogDifferentiablePolicy {

	/** the policy parameters */
	protected double[] theta;
	
	/**
	 * Construct a {@link LogDifferentiablePolicy}.
	 * @param n length of the parameters vector
	 */
	public LogDifferentiablePolicy(int n) {
        this.theta = new double[n];
	}
    
    /* (non-Javadoc)
     * @see jrl.utils.ParametricFunction#boundParams(double[])
     */
    public boolean boundParams(@NonNull final double[] params) {
    	// No bounds by default
		return false;
    }
    
    /* (non-Javadoc)
     * @see jrl.utils.ParametricFunction#getParams()
     */
    public @NonNull final double[] getParams() {
		return theta;
	}
    
    /* (non-Javadoc)
     * @see jrl.utils.ParametricFunction#getParamsSize()
     */
    public final int getParamsSize() {
    	return theta.length;
    }

    /* (non-Javadoc)
     * @see jrl.utils.ParametricFunction#setParams(double[])
     */
    public void setParams(@NonNull final double[] theta) {
    	System.arraycopy(theta, 0, this.theta, 0, theta.length);
	}
    
    /* (non-Javadoc)
     * @see jrl.utils.ParametricFunction#updateParams(double[])
     */
    public void updateParams(@NonNull final double[] delta) {
    	if(theta.length != delta.length) {
    		throw new IllegalArgumentException("Incorrect vector length");
    	}
    	// assert theta.length == delta.length
    	for(int i=0; i<theta.length; i++) {
    		theta[i] = theta[i] + delta[i];
    	}
    	boundParams( theta );
    }

}
