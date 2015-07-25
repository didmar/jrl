package com.github.didmar.jrl.environment.dynsys;

/**
 * A dynamic system can be seen as a continuous-time and deterministic
 * transition function: given a current state and action, it computes the time
 * derivative of the state.
 * @author Didier Marin
 */
public interface DynamicSystem {
	/**
	 * Compute the dynamics of the system, given the current state and action.
	 * @param x
	 * @param u
	 * @param dotX
	 */
	public void dotX(double[] x, double[] u, double[] dotX);
	/**
	 * Compute ∂dotX(x,u)/∂x, the partial derivative of the dynamics with respect to the state.
	 * @param x
	 * @param u
	 * @param derX	[on return] a <tt>xDim</tt>-by-<tt>xDim</tt> matrix such that
	 * 				derX[i][j] = ∂dotX(x,u)[i]/x[j]
	 */
	public void derX(double[] x, double[] u, double[][] derX);
	/**
	 * Compute ∂dotX(x,u)/∂u, the partial derivative of the dynamics with respect to the action.
	 * @param x
	 * @param u
	 * @param derU	[on return] a <tt>xDim</tt>-by-<tt>uDim</tt> matrix such that
	 * 				derU[i][j] = ∂dotX(x,u)[i]/u[j]
	 */
	public void derU(double[] x, double[] u, double[][] derU);

	public int getXDim();

	public int getUDim();
}
