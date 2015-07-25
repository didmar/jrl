package com.github.didmar.jrl.evaluation.valuefunction;

import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * State action value function represented by a state card-by-action card
 * matrix.
 * @author Didier Marin
 */
public final class TabularQFunction extends QFunction {

	private final double[][] Q;

	public TabularQFunction(int xCard, int uCard) {
		Q = ArrUtils.zeros(xCard, uCard);
	}

	public TabularQFunction(double[][] Q) {
		this.Q = ArrUtils.cloneMatrix(Q, Q.length, Q[0].length);
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.valuefunction.QFunction#get(double[], double[])
	 */
	@Override
	public final double get(double[] x, double[] u) {
		return Q[(int)x[0]][(int)u[0]];
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.valuefunction.QFunction#updateForStateAction(double[], double[], double)
	 */
	@Override
	public final void updateForStateAction(double[] x, double[] u, double delta) {
		Q[(int)x[0]][(int)u[0]] += delta;
	}

	@Override
	public final int getXDim() {
		return 1;
	}

	@Override
	public final int getUDim() {
		return 1;
	}

}
