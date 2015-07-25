package com.github.didmar.jrl.features;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;

/**
 * Compatible features according to Sutton's Policy Gradient Theorem :
 * these state-action features correspond to the gradient of the log of
 * a {@link ILogDifferentiablePolicy}.
 *
 * @see ILogDifferentiablePolicy
 * @author Didier Marin
 */
public final class CompatibleFeatures extends Features {

	/** The compatible policy from which to get the features */
	protected final ILogDifferentiablePolicy pol;
	/** State-space dimension */
	protected final int xDim;
	/** Action-space dimension */
	protected final int uDim;
	/** Number of policy parameters in the policy
	 * from which we get the features*/
	protected final int n;

	// arrays for temporary storage to avoid mem. alloc.
	private final double[] xTmp;
	private final double[] uTmp;

	public CompatibleFeatures(ILogDifferentiablePolicy pol, int xDim, int uDim){
		super(xDim+uDim, pol.getParamsSize());
		this.pol = pol;
		this.xDim = xDim;
		this.uDim = uDim;
		n = pol.getParamsSize();

		// arrays for temporary storage to avoid mem. alloc.
		xTmp = new double[xDim];
		uTmp = new double[uDim];
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#phi(double[], double[])
	 */
	@Override
	public final void phi(double[] x, double[] y) {
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		if(y.length != outDim ){
			throw new IllegalArgumentException("y must have length outDim");
		}

		// x contains both the state and the action since compatible features
		// are state-action features, so we first extract these two
		System.arraycopy(x, 0, xTmp, 0, xDim);
		System.arraycopy(x, xDim, uTmp, 0, uDim);
		// Get the gradient of the log of the given policy
		// and store it in y
		System.arraycopy(pol.dLogdTheta(xTmp,uTmp),0,y,0,n);
	}

	@Override
	public final boolean isNormalized() {
		return false;
	}

}
