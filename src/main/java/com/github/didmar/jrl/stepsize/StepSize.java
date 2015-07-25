package com.github.didmar.jrl.stepsize;

/**
 * A step-size is a set of parameters that evolve through time.
 * @author Didier Marin
 *
 */
public interface StepSize {
	
	/**
	 * Update the step-size based on an internal counter
	 */
	public void updateStep();
	
	/**
	 * Set the step-size based on a time step
	 */
	public void stepAtTime(int t);
	
	/**
	 * Returns the current step
	 */
	public double getStep();
}
