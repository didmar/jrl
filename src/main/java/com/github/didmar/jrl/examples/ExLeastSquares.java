package com.github.didmar.jrl.examples;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.features.FourierRandomFeatures;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.regression.LeastSquares;

/**
 * Testing {@link LeastSquares}.
 * 
 * @author Didier Marin
 */
public class ExLeastSquares {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int inDim = 1;
		int nbFeatures = 10;
		double sigma = 1.;
		FourierRandomFeatures feat = new FourierRandomFeatures(inDim, nbFeatures, sigma);
		double[] optParams = ArrUtils.rand(nbFeatures);
		
		double regularization = 0.;
		LeastSquares ls = new LeastSquares(feat,regularization);
		int nbSamples = 1000;
		int testSteps = 100;
		
		double[] x = new double[inDim];
		double[] phiX = new double[nbFeatures];
		
		// Learning
		for (int i = 1; i <= nbSamples; i++) {
			// Create a random input
			ArrUtils.rand(x);
			// Compute the features for this input
			feat.phi(x,phiX);
			// Compute the "optimal" output
			double y = ArrUtils.dotProduct(phiX, optParams, nbFeatures);
			// Feed the Least Squares
			ls.addSample(x, y);
			if(i % 10 == 0) {
				ls.computeParameters();
				double[] errors = testPredictor(inDim, ls, optParams, testSteps);
				System.out.println("i="+i+" : MSE="+ArrUtils.mean(errors));
			}
		}
		
		System.out.println("optParams    ="+ArrUtils.toString(optParams));
		System.out.println("learnedParams="+ArrUtils.toString(ls.getParams()));
	}
	
	public static double[] testPredictor(int inDim, LeastSquares ls,
			double[] optParams, int steps) {
		Features feat = ls.getFeatures();
		int nbFeatures = feat.outDim;
		double[] phiX = new double[nbFeatures];
		double[][] gridX = ArrUtils.buildGrid(ArrUtils.zeros(inDim), ArrUtils.ones(inDim), steps);
		int nbTestSamples = gridX.length;
		double[] errors = new double[nbTestSamples]; 
		for (int i = 0; i < nbTestSamples; i++) {
			// Compute the features for this input
			feat.phi(gridX[i],phiX);
			// Compute the "optimal" output
			double y = ArrUtils.dotProduct(phiX, optParams, nbFeatures);
			// Compute the predicted output
			double hatY = ls.predict(gridX[i]);
			// Compute the error
			errors[i] = Math.pow(hatY - y, 2);
		}
		return errors;
	}

}
