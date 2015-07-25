package com.github.didmar.jrl.features;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Features are vector functions of an real input space, which is
 * generally a state or the state-action space.
 * They are the basis for function approximators such as
 * {@link jrl_testing.evaluation.valuefunction.LinearVFunction}.
 * 
 * @author Didier Marin
 */
public abstract class Features {
	
	@Override
	public @NonNull String toString() {
		return "Features("+inDim+","+outDim+")";
	}

	/** Input dimension of the features function */
    public final int inDim;
    /** Output dimension of the features function */
    public final int outDim;

    /**
     * @param inDim input dimension of the features function
     * @param outDim ouput dimension of the features function
     * @param IllegalArgumentException when dimensions are invalid
     */
	public Features(int inDim, int outDim)
			throws IllegalArgumentException {
		if(inDim <= 0) {
			throw new IllegalArgumentException("inDim must be greater than 0");
		}
		if(outDim <= 0) {
			throw new IllegalArgumentException("outDim must be greater than 0");
		}
		this.inDim = inDim;
		this.outDim = outDim;
    }
    
	/**
	 * Compute the features vector for a given input and return the result
	 * in a new vector.
	 * @param x the input
	 * @return the output in a new vector
	 * @throws IllegalArgumentException when the value x is invalid
	 */
	public final double[] phi(final double[] x)
			throws IllegalArgumentException {
		assert x != null;
		
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		
    	final double[] y = new double[outDim];
    	phi(x,y);
    	
    	assert (!isNormalized()) || Utils.allClose(ArrUtils.sum(y),1.,Utils.getMacheps())
    		: "Features should be normalized";
    	
    	return y;
    }
    
    /**
     * Compute the features vector for a given input
     * and store the result in a given array.
     * @param x an array containing the input
     * @param y an array to store the ouput
     */
    public abstract void phi(final double[] x, final double[] y);
    
    /**
	 * Returns whether the output components are always positive and sums to 1.
	 * @return true if the output components are always positive and sums to 1,
	 *         false else
	 */
	public abstract boolean isNormalized();
}
