package com.github.didmar.jrl.utils;

/**
 * This interface represents the real vector parameters of a function, which
 * components might be bounded. The length
 * @author Didier Marin
 */
public interface ParametricFunction {

	/**
	 * Returns the parameters.
	 */
	public double[] getParams();

	/**
	 * Returns the number of parameters.
	 */
	public int getParamsSize();

	/**
	 * Set the parameters by copy.
	 * Precondition : the new parameters must be within the bounds, if any.
	 * @see ParametricFunction#boundParams(double[])
	 */
	public void setParams(final double[] params);
	
	/**
	 * Update the parameters vector by adding a vector delta to it.
	 * Precondition : delta size must equals parameters vector size.
	 * {@link ParametricFunction#getParamsSize()}
	 * @param delta    real vector to add to the parameters vector
	 */
	public void updateParams(final double[] delta);

	/**
	 * Bound the given parameters.
	 * @return true if some of the parameters were out of bounds, false else
	 */
	public boolean boundParams(final double[] params);

}