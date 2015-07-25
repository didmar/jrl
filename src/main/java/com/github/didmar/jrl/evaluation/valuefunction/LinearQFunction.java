package com.github.didmar.jrl.evaluation.valuefunction;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * State-action value function based on a linear combination of state-action
 * features.
 * @author Didier Marin
 */
public final class LinearQFunction extends ParametricQFunction {

	/** State-action space features */
	private final Features features;
	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;

	// arrays for temporary storage to avoid mem. alloc.
	private final double[] xu;
	private final double[] deltaTmp;

	public LinearQFunction(Features features, int xDim, int uDim) {
		super(new double[features.outDim]);
		if(features.inDim != xDim + uDim) {
			throw new IllegalArgumentException(
					"Invalid state-action features input dimension");
		}
		this.features = features;
		this.xDim = xDim;
		this.uDim = uDim;

		// arrays for temporary storage to avoid mem. alloc.
		xu = new double[this.xDim + this.uDim];
		deltaTmp = new double[n];
	}

	public LinearQFunction(Features features, int xDim, int uDim,
			double[] params) {
		super(params);
		if(getParamsSize() != features.outDim) {
			throw new IllegalArgumentException("Mismatch between parameters vector"
				+ " length and features output dim");
		}
		if(features.inDim != xDim + uDim) {
			throw new IllegalArgumentException(
					"Invalid state-action features input dimension");
		}
		this.features = features;
		this.xDim = xDim;
		this.uDim = uDim;

		// arrays for temporary storage to avoid mem. alloc.
		xu = new double[this.xDim + this.uDim];
		deltaTmp = new double[n];
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.QFunction#get(double[],double[])
	 */
	public final double get(double[] x, double[] u) {
		assert(x.length == xDim);
		assert(u.length == uDim);

		System.arraycopy(x,0,xu,0,xDim);
		System.arraycopy(u,0,xu,xDim,uDim);
		return ArrUtils.dotProduct(features.phi(xu), getParams(), getParamsSize());
	}

	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#boundParams(double[])
	 */
	public final boolean boundParams(double[] params) {
		// TODO add large bounds
		return false;
	}

	/* (non-Javadoc)
	 * @see jrl.evaluation.valuefunction.QFunction#updateForStateAction(double[], double[], double)
	 */
	public final void updateForStateAction(double[] x, double[] u, double delta) {
		System.arraycopy(x,0,xu,0,xDim);
		System.arraycopy(u,0,xu,xDim,uDim);
		final double[] phixu = features.phi(xu);
		updateForFeaturesVector(phixu, delta);
	}

	/**
	 * Perform a parameter update of norm delta in direction phixu.
	 * @param phixu  direction of the update in the parameters space
	 * @param delta  norm of the update in the parameters space
	 */
	public final void updateForFeaturesVector(double[] phixu, double delta) {
		for(int i=0; i<deltaTmp.length; i++) {
			deltaTmp[i] = delta * phixu[i];
		}
		updateParams(deltaTmp);
	}

	public final Features getFeatures() {
		return features;
	}

	@Override
	public final int getXDim() {
		return xDim;
	}

	@Override
	public final int getUDim() {
		return uDim;
	}

}
