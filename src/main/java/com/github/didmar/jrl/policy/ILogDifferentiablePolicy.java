package com.github.didmar.jrl.policy;

//TODO add a method to ILogDifferentiablePolicy that generates compatible features,
//     and implement it in LogDifferentiablePolicy
/**
 * A parametric policy that is log-differentiable with respect to its
 * parameters.
 * @author Didier Marin
 */
public interface ILogDifferentiablePolicy extends ParametricPolicy {
	
	/**
	 * Returns the derivative of the log policy over the parameters,
	 * given the state and the action.
	 */
    public double[] dLogdTheta(double[] x, double[] u);
}
