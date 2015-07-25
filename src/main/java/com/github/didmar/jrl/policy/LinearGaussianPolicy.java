package com.github.didmar.jrl.policy;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * A Gaussian policy which mean is a sum of normalized state features weighted
 * by the policy parameters (hence the linear term). By default, the mean action
 * is bounded and the policy parameters are normalized such that 0 correspond to
 * the minimal action and 1 to the maximal action.
 * @author Didier Marin
 */
public final class LinearGaussianPolicy extends GaussianPolicy {

	/** Normalized state features */
	private final Features stateFeatures;
	/** Number of normalized state features */
	private final int nFeat;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Used to store features for a given state */
	private final double[] phix;
	
	/** If true, the policy parameters will be bounded using the boundedParams
	 * array
	 */
	private boolean boundedParams;
	
	/**
	 * Bounds for the policy parameters space. First dimension is the
	 * parameters dimension, second dimension is 2 : min and max.
	 */
	private final double[][] thetaBounds;
	
	private boolean normalizedActions;

	/**
	 * Construct a {@link LinearGaussianPolicy}.
	 * @param stateFeatures    normalized state features
	 * @param sigma            std dev of the policy normal distribution
	 * @param uBounds          bounds of the action
	 * @throws Exception 
	 */
	public LinearGaussianPolicy(final Features stateFeatures,
								final double[] sigma,
								final double[] uMin,
								final double[] uMax,
								boolean normalizedActions) {
		super(stateFeatures.inDim, sigma, uMin, uMax, stateFeatures.outDim*sigma.length);
		if(normalizedActions && !stateFeatures.isNormalized()) {
			throw new IllegalArgumentException("If using normalized actions, state features"
					+" should be normalized");
		}
		this.stateFeatures = stateFeatures;
		this.normalizedActions = normalizedActions;
        nFeat = stateFeatures.outDim;
        
        // Use 0.5 as default parameters, which correspond to the mean
        // between the default bounds.
        setParams(ArrUtils.constvec(n,0.5));
        
        // Set the default parameters bounds : [0.,1.] for every parameter
        boundedParams = true;
        thetaBounds = new double[theta.length][2];
        setParamsBounds(0., 1.);
        
        // arrays for temporary storage to avoid mem alloc.
        phix = new double[nFeat];
	}
	
	@Override
    public final void meanAction(@NonNull final double[] x,
    							 @NonNull final double[] mu) {
		assert(x.length == xDim);
		assert(mu.length == uDim);
        // Compute the normalized state features
        stateFeatures.phi(x,phix);
        // Compute the mean action (un-normalized)
        for(int i=0; i<uDim; i++) {
        	double phixTheta = 0.;
        	for(int j=0; j<nFeat; j++) {
        		phixTheta += phix[j] * theta[i*nFeat+j];
        	}
        	if(normalizedActions) {
        		// unnormalize phixTheta to obtain the mean action
        		mu[i] = uMin[i] + (uMax[i] - uMin[i]) * phixTheta;
        	} else {
        		// else, use phixTheta as the mean action
        		mu[i] = phixTheta;
        	}
        }
	}
	
	@Override
	public final void dMeanActiondTheta(@NonNull final double[] x,
										@NonNull final double[] dermu) {
		// For each state feature, compute the derivative
		for(int i=0; i<nFeat; i++) {
			dermu[i] = phix[i];
		}
	}
	
	@Override
	public final boolean boundParams(@NonNull final double[] params) {
		if(this.theta.length != params.length) {
			throw new RuntimeException("Incorrect vector length");
		}
		if(boundedParams) {
			// Bound the parameters
			boolean bounded = false;
			for(int i=0; i<params.length; i++) {
				if(params[i] < thetaBounds[i][0]) {
					params[i] = thetaBounds[i][0];
					bounded = true;
				} else if(params[i] > thetaBounds[i][1]) {
						params[i] = thetaBounds[i][1];
						bounded = true;
				}
			}
			return bounded;
		}
		return true;
    }
	
	/**
	 * A convenience function to set the parameters for each action dimension.
	 * @param params an array of uDim-by-nFeat
	 */
	public final void setParamsByActionDim(final double[][] params) {
		if(params.length != uDim) {
			throw new IllegalArgumentException("params matrix must have uDim rows");
		}
		for (int i = 0; i < uDim; i++) {
			if(params[i].length != nFeat) {
				throw new IllegalArgumentException("params matrix must have nFeat columns");
			}
			for(int j=0; j<nFeat; j++) {
	        	theta[i*nFeat+j] = params[i][j];
	        }
		}
	}

	public final void useBoundedParams(boolean use) {
		boundedParams = use;
	}
	
	public final void useNormalizedActions(boolean use) {
		if(use && !stateFeatures.isNormalized()) {
			throw new IllegalArgumentException("If using normalized actions, state features"
					+" should be normalized");
		}
		normalizedActions = use;
	}
	
	public final void setParamsBounds(final double[][] thetaBounds) {
		// TODO check the size of thetaBounds and that min < max for all params
		ArrUtils.copyMatrix(thetaBounds, this.thetaBounds, n, 2);
	}
	
	public final void setParamsBounds(double thetaMin, double thetaMax) {
		assert thetaMin <= thetaMax;
		
		if(normalizedActions && (thetaMin != 0. || thetaMax != 1.)) {
			throw new IllegalArgumentException("If using normalized actions,"
					+" policy parameters should be bounded in [0,1]");
		}
		for (int i = 0; i < thetaBounds.length; i++) {
			thetaBounds[i][0] = thetaMin;
			thetaBounds[i][1] = thetaMax;
		}
	}
	
	@Override
	@NonNull
	public final String toString() {
		return "LinearGaussianPolicy";
	}
}
