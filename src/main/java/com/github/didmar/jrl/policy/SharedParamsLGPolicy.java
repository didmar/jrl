package com.github.didmar.jrl.policy;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.policy.GaussianPolicy;
import com.github.didmar.jrl.policy.LinearGaussianPolicy;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * A variant of LinearGaussianPolicy where some parameters are shared, i.e.,
 * they weight more than one feature. The chosen number of parameters must be
 * dividable by the action-space dimension ! For instance, for a 2-dim action
 * space, 6 parameters will splitted in 3 for each action dimension. If the
 * features output is 6-dim, each of the 3 params will weight 2 features.
 * @author Didier Marin
 */
public final class SharedParamsLGPolicy extends GaussianPolicy {

	/** Normalized state features */
	private final Features stateFeatures;
	/** Number of normalized state features */
	private final int nFeat;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Used to store features for a given state */
	private final double[] phix;

	/**
	 * Construct a {@link LinearGaussianPolicy}.
	 * @param stateFeatures    normalized state features
	 * @param sigma            std dev of the policy normal distribution
	 * @param uBounds          bounds of the action
	 * @throws Exception 
	 */
	public SharedParamsLGPolicy(Features stateFeatures,	double[] sigma,
			double[] uMin, double[] uMax, int nbParams) throws Exception {
		super(stateFeatures.inDim, sigma, uMin, uMax, nbParams);
		assert( n % uDim == 0 );
		// Check that features are normalized
		if(!stateFeatures.isNormalized()) {
			throw new Exception("Features must be normalized");
		}
		this.stateFeatures = stateFeatures;
		nFeat = stateFeatures.outDim;
        assert( nFeat % (n/uDim) == 0 );
        
        // Use 0.5 as default parameters, which correspond to the mean
        // between the minimal and maximal action.
        setParams(ArrUtils.constvec(n,0.5));
        
        // arrays for temporary storage to avoid mem alloc.
        phix = new double[nFeat];
	}
	
	@Override
    public final void meanAction(double[] x, double[] mu) {
        // Compute the normalized state features
        stateFeatures.phi(x,phix);
        // Compute the mean action (un-normalized)
        for(int i=0; i<uDim; i++) {
        	double phixTheta = 0.;
        	for(int j=0; j<nFeat; j++) {
        		phixTheta += phix[j] * theta[i*(n/uDim)+j/(nFeat/(n/uDim))];
        	}
            mu[i] = uMin[i] + (uMax[i] - uMin[i]) * phixTheta;
        }
	}
	
	@Override
	public final void dMeanActiondTheta(double[] x, double[] dermu) {
		// TODO implement me !
		throw new UnsupportedOperationException();
	}
	
	@Override
	public final boolean boundParams(double[] theta) {
		if(this.theta.length != theta.length) {
			throw new RuntimeException("Incorrect vector length");
		}
		// Bound the parameters in [0,1]
		boolean bounded = false;
		for(int i=0; i<theta.length; i++) {
			if(theta[i] < 0.) {
				theta[i] = 0.;
				bounded = true;
			} else {
				if(theta[i] > 1.) {
					theta[i] = 1.;
					bounded = true;
				}
			}
		}
		return bounded;
    }

	@Override
	public final String toString() {
		return "SharedParamsLinearGaussianPolicy";
	}
}
