package com.github.didmar.jrl.evaluation.valuefunction;

/**
 * State value function to which a state value function is added as a baseline. 
 * @author Didier Marin
 */
public final class QFunctionWithBaseline extends QFunction {

	private final QFunction qFunction;
	private final VFunction vFunction;
	private final int sign;

	/**
	 * Construct a {@link QFunctionWithBaseline}
	 * @param qFunction   the state action value function to add a baseline to
	 * @param vFunction   the state value function baseline
	 * @param sign        +1 to add the baseline, -1 to substract it
	 */
	public QFunctionWithBaseline(QFunction qFunction, VFunction vFunction,
			int sign) {
		this.qFunction = qFunction;
		this.vFunction = vFunction;
		this.sign = sign;
	}
	
	/* (non-Javadoc)
	 * @see jrl.evaluation.valuefunction.QFunction#get(double[], double[])
	 */
	@Override
	public final double get(double[] x, double[] u) {
		return qFunction.get(x, u) + sign*vFunction.get(x);
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.valuefunction.QFunction#updateForStateAction(double[], double[], double)
	 */
	@Override
	public final void updateForStateAction(double[] x, double[] u, double delta) {
		qFunction.updateForStateAction(x, u, delta);
	}

	@Override
	public final int getXDim() {
		return qFunction.getXDim();
	}

	@Override
	public final int getUDim() {
		return qFunction.getUDim();
	}

}
