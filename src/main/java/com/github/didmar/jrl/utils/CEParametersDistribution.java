package com.github.didmar.jrl.utils;

import com.github.didmar.jrl.utils.GaussianParametersDistribution;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Cross-Entropy method that computes a prior distribution over parameters.
 * @author Didier Marin
 */
public final class CEParametersDistribution
		extends GaussianParametersDistribution {
	
	public CEParametersDistribution(double[] mean, double[] sigma) {
		super(mean, sigma);
	}
	
	public CEParametersDistribution(double[] R,	double[][] params,
			int nSelected, double noise) {
		super(new double[params[0].length], new double[params[0].length]);
		computeParamsDistribution(R,params,nSelected,noise,false);
	}
	
	/**
	 * Compute the CE distribution as the mean and std dev of the nSelected best
	 * parameter vectors, according to the performance mesure R.
	 * Returns the index, in parameter vectors array, of the best parameter
	 * vector.
	 *  
	 * @param R         array of performance of the parameter vectors
	 * @param params    array of parameter vectors
	 * @param nSelected number of the best parameter vectors to use as
	 *                  samples of the new distribution
	 * @param noise     supplementary term added to sigma in order to avoid
	 *                  premature convergence 
	 * @param greedy    if true, use the best parameters as the new mean and 
	 *                  the supplementary noise term as the new sigma.
	 * @return index of the best parameter vector in params
	 */
	public final int computeParamsDistribution(double[] R, double[][] params,
			int nSelected, double noise, boolean greedy) {
		// assert R.length == thetas.length
		// assert nSelected > 0;
		final int nParams = R.length;
		// Check the dimensions of params
		if(nParams != params.length) {
			throw new RuntimeException("Incorrect array length");
		}
		for(int i=0; i<nParams; i++) {
			if(params[i].length != n) {
        		throw new RuntimeException("Incorrect array length");
        	}
		}
		// Sort the parameters vector by their performance
        final int[] index = new int[nParams];
        Utils.quicksort(R, index);
        if(greedy) {
        	if(params[index[nParams-1]].length != n) {
        		throw new RuntimeException("Incorrect array length");
        	}
        	System.arraycopy(params[index[nParams-1]], 0, mean, 0, n);
        	ArrUtils.constvec(sigma, noise);
        } else {
	        // Compute the mean and std dev of the nSelected best parameters
	        double[] tmp = new double[nSelected];
	        for(int i=0; i<n; i++) {
	        	for(int j=0; j<nSelected; j++) {
	        		tmp[j] = params[index[nParams-1-j]][i];
	        	}
	        	mean[i] = ArrUtils.mean(tmp);
	        	sigma[i] = ArrUtils.std(tmp) + noise;
	        }
        }
        return index[nParams-1];
	}
}
