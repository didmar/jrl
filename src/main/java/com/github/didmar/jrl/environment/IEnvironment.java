package com.github.didmar.jrl.environment;

import org.eclipse.jdt.annotation.Nullable;

import com.github.didmar.jrl.agent.Agent;

/**
 * An environment represents a Markovian Decision Process that can be used in
 * interaction with an agent.
 * @see Environment
 * @author Didier Marin
 */
public interface IEnvironment extends Iterable<EnvironmentListener> {

	/**
	 * Returns an initial state.
	 * @return an initial state
	 */
	public double[] startState();

	/**
	 * Returns the next state given a state and an action
	 * @param x    a state
	 * @param u    an state
	 * @return the next state
	 */
	public double[] nextState(double[] x, double[] u);

	/**
	 * Returns the reward given a state, an action and a next state. Note that
	 * the next state might not be necessary in some cases.
	 * @param x    a state
	 * @param u    an action
	 * @param xn   a next state
	 * @return the reward
	 */
	public double reward(double[] x, double[] u, double[] xn);

	/**
	 * Returns whether a state, action and next state sample is terminal or not.
	 * @param x    a state
	 * @param u    an action
	 * @param xn   a next state
	 * @return true is (x,u,xn) is a terminal sample, false else
	 */
	public boolean isTerminal(double[] x, double[] u, double[] xn);

	/**
	 * Raise an exception if the given action is illegal
	 * @throws Exception
	 */
	public void checkIfLegalAction(double[] u) throws Exception;

	/**
	 * Make an agent interact with the environment.
	 * @param agent the agent to interact with
	 * @param nbEpi the number of episodes to perform
	 * @param maxT  the maximum duration of an episode
	 * @param x0    the initial state (optional)
	 */
	public void interact(Agent agent, int nbEpi, int maxT, @Nullable double[] x0);

	/**
     * Make an agent interact with the environment, without specifying the start
     * state.
     * @param agent the agent to interact with
     * @param nbEpi the number of episodes to perform
     * @param maxT  the maximum duration of an episode
     */
	public void interact(Agent agent, int nbEpi, int maxT);
	
	/**
	 * Add a listener to the environment.
	 * @param listener the listener to be added
	 */
	public void addListener(EnvironmentListener listener);

	/**
	 * Remove a listener from the environment.
	 * @param listener the listener to be removed
	 * @return true if the environment contained the listener
	 */
	public boolean removeListener(EnvironmentListener listener);
	
	/**
	 * Remove all the listeners from the environment.
	 */
	public void removeAllListener();
	
	/**
	 * Returns the state-space dimension.
	 * @return the state-space dimension
	 */
	public int getXDim();

	/**
	 * Returns the action-space dimension.
	 * @return the action-space dimension
	 */
	public int getUDim();
	
	/**
	 * Returns the minimum of each state component, or <code>null</code> if
	 * it is not bounded.
	 * @return the minimum of each state component, or <code>null</code> if
	 * 		   it is not bounded
	 */
	public @Nullable double[] getXMin();
	
	/**
	 * Returns the maximum of each state component, or <code>null</code> if
	 * it is not bounded.
	 * @return the maximum of each state component, or <code>null</code> if
	 * 		   it is not bounded
	 */
	public @Nullable double[] getXMax();

	/**
	 * Returns the minimum of each action component, or <code>null</code> if
	 * it is not bounded.
	 * @return the minimum of each action component, or <code>null</code> if
	 * 		   it is not bounded
	 */
	public @Nullable double[] getUMin();

	/**
	 * Returns the maximum of each action component, or <code>null</code> if
	 * it is not bounded.
	 * @return the maximum of each action component, or <code>null</code> if
	 * 		   it is not bounded
	 */
	public @Nullable double[] getUMax();

}