package com.github.didmar.jrl.evaluation.valuefunction;

/**
 * State action value function represented by a state card-by-action card
 * matrix.
 * @author Didier Marin
 */
public final class TabularVFunction extends VFunction {

	private final double[] V;

	public TabularVFunction(double[] V) {
		this.V = V.clone();
	}
	
	@Override
	public final double get(double[] x) {
		return  V[(int)x[0]];
	}

	@Override
	public final void updateForState(double[] x, double delta) {
		V[(int)x[0]] += delta;
	}

	@Override
	public final int getXDim() {
		return 1;
	}
	
}
