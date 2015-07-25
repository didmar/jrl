package com.github.didmar.jrl.evaluation.valuefunction;

import com.github.didmar.jrl.utils.DiscountFactor;

/**
 * Abstract class that represents a state-action value function or an advantage
 * function.
 * @author Didier Marin
 */
public abstract class QFunction {
	
	/**
	 * Returns the state-action value for state x and action u.
	 * @param x the state from the state-action pair to get the value of
	 * @param u the action from the state-action pair to get the value of
	 * @return the state value in state x
	 */
	public abstract double get(double[] x, double[] u);
	
	/**
	 * Update the value for state x and action u by increment delta.
	 * @param x     the state from the state-action pair to update the value of
	 * @param u     the action from the state-action pair to update the value of
	 * @param delta the increment to add to the current value for state x and 
	 *              action u 
	 */
	public abstract void updateForStateAction(double[] x, double[] u,
			double delta);
	
	// TODO add in arguments a boolean that indicates if the sample is terminal,
	// and if it is don't add the gamma*get(xn,un) term
	/**
	 * Returns the Temporal Difference (TD) error between the state-action pair
	 * (x,u) and the next state-action pair (xn,un) given reward r, in a
	 * discounted reward setting. This TD error is used by {@link SARSA}.
	 * @param x     a state
	 * @param u     an action
	 * @param xn    a next state
	 * @param un    a next action
	 * @param r     a reward
	 * @param isTerminal indicates whether the tuple (x,u,xn) is terminal or not
	 * @param gamma a discount factor
	 * @return the TD error between state-action pair (x,u) and next
	 * state-action pair (xn,un) given reward r
	 */
	public final double tdError(double[] x, double[] u, double[] xn,
			double[] un, double r, boolean isTerminal, DiscountFactor gamma) {
		if(isTerminal) {
			return r - get(x,u);
		}
		return r + gamma.value*get(xn,un) - get(x,u);
	}
	
	/**
	 * Returns the state-space dimension. 
	 * @return the state-space dimension.
	 */
	public abstract int getXDim();
	
	/**
	 * Returns the action-space dimension. 
	 * @return the action-space dimension.
	 */
	public abstract int getUDim();
}
