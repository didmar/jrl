package com.github.didmar.jrl.agent;

/**
 * An agent is something that interact with an environment by taking actions
 * depending on the state of this environment.
 * 
 * @author Didier Marin
 */
public interface Agent {
	
	/**
	 * Returns the action chosen by the agent given the state x
	 * @param x   a state of the environment
	 * @return the action chosen by the agent given the state x
	 */
	public double[] takeAction(double[] x);
}
