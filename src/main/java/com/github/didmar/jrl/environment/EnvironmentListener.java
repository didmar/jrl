package com.github.didmar.jrl.environment;

/**
 * An EnvironmentListener is notified of the interactions (beginning of an
 * episode, new step and end of an episode) that occurs in the environment
 * its listening to.
 * @see Environment
 * @author Didier Marin
 */
public interface EnvironmentListener {
	
	/**
	 * Notify the beginning of a new episode
	 * @param x0    the start state
	 * @param maxT  the maximum length of this episode
	 */
	public void newEpisode(double[] x0, int maxT);
	
	/**
	 * Notify a (x,u,xn,r) sample
	 * @param x    the state
	 * @param u    the action
	 * @param xn   the next state
	 * @param r    the reward
	 * @param isTerminal indicates whether if the tuple (x,u,xn) is terminal or not
	 */
	public void receiveSample(double[] x, double[] u, double[] xn,
							  double r, boolean isTerminal);
	
	/**
	 * Notify the end of an episode
	 */
	public void endEpisode();
}
