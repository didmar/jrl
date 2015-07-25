package com.github.didmar.jrl.evaluation.valuefunction;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * State value function based on a linear combination of state features.
 * @author Didier Marin
 */
public class LinearVFunction extends ParametricVFunction {

	/** State-space features */
	private final Features features;

	// arrays for temporary storage to avoid mem. alloc.
	private final double[] deltaTmp;

	public LinearVFunction(Features features) {
		super(ArrUtils.zeros(features.outDim));
		this.features = features;

		// arrays for temporary storage to avoid mem. alloc.
		deltaTmp = new double[n];
	}

	public LinearVFunction(Features features, double[] params) {
		super(params);
		if(getParamsSize() != features.outDim) {
			throw new IllegalArgumentException("Mismatch between parameters vector"
				+ " length and features output dim");
		}
		this.features = features;

		// arrays for temporary storage to avoid mem. alloc.
		deltaTmp = new double[n];
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.VFunction#get(double[])
	 */
	public final double get(double[] x) {
		return ArrUtils.dotProduct(features.phi(x), getParams(), getParamsSize());
	}

	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#boundParams(double[])
	 */
	public final boolean boundParams(double[] params) {
		// TODO add large bounds
		return false;
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.valuefunction.VFunction#updateForState(double[], double)
	 */
	public final void updateForState(double[] x, double delta) {
		final double[] phix = features.phi(x);
		updateForFeaturesVector(phix, delta);
	}

	/**
	 * Perform a parameter update of norm delta in direction phix.
	 * @param phix  direction of the update in the parameters space
	 * @param delta norm of the update in the parameters space
	 */
	public final void updateForFeaturesVector(double[] phix, double delta) {
		for(int i=0; i<deltaTmp.length; i++) {
			deltaTmp[i] = delta * phix[i];
		}
		updateParams(deltaTmp);
	}

	public final Features getFeatures() {
		return features;
	}

	@Override
	public final int getXDim() {
		return features.inDim;
	}

	@Override
	public final Object clone() throws CloneNotSupportedException {
		LinearVFunction cl = new LinearVFunction(features);
		cl.setParams(getParams());
		return cl;
	}


}
