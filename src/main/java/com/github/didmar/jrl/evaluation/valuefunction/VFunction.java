package com.github.didmar.jrl.evaluation.valuefunction;

import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * Interface that represents a state value function.
 * @author Didier Marin
 */
public abstract class VFunction {
	
	/**
	 * Returns the state value in state x.
	 * @param x the state to get the value of
	 * @return the state value in state x
	 */
	public abstract double get(double[] x);
	
	/**
	 * Update the value for state x by increment delta.
	 * @param x     the state to update the value of
	 * @param delta the increment to add to the current value of state x 
	 */
	public abstract void updateForState(double[] x, double delta);
	
	// TODO add in arguments a boolean that indicates if the sample is terminal,
	// and if it is don't add the gamma*get(xn) term
	/**
	 * Returns the Temporal Difference (TD) error between state x and next state
	 * xn given reward r, in a discounted reward setting.
	 * @param x     a state
	 * @param xn    a next state
	 * @param r     a reward
	 * @param gamma a discount factor
	 * @param isTerminal indicates if the tuple (x,u,xn) is terminal
	 * @return the TD error between state x and next state xn given reward r
	 */
	public final double tdError(double[] x, double[] xn, double r,
			boolean isTerminal, DiscountFactor gamma) {
		if(isTerminal) {
			return r - get(x);
		}
		return r + gamma.value * get(xn) - get(x);
	}
	
	/**
	 * Returns the state-space dimension. 
	 * @return the state-space dimension.
	 */
	public abstract int getXDim();
}
