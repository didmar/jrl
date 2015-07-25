package com.github.didmar.jrl.examples;

import java.io.IOException;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.features.FourierRandomFeatures;
import com.github.didmar.jrl.features.RBFFeatures;
import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;
import com.github.didmar.jrl.utils.plot.Plot2D;

// TODO plot all component at once
/**
 * Display some of the implemented {@link com.github.didmar.jrl.features.Features}.
 * 
 * @see com.github.didmar.jrl.features
 * @author Didier Marin
 */
public class ExFeatures {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		final int xDim = 1;
		@NonNull final double[] xMin = ArrUtils.zeros(xDim);
		@NonNull final double[] xMax = ArrUtils.ones(xDim);
		
		int nbSamplesPerDim = 101;
		//final double[][] xSamples = ArrUtils.buildGrid(xMin, xMax, nbSamplesPerDim);
		@NonNull final double[] xSamples =
				ArrUtils.linspace(xMin[0], xMax[0], nbSamplesPerDim);
		
		final int featType = Utils.chooseOne(
				new String[]{"RBFFeatures","FourierRandomFeatures"});
		
		switch(featType) {
			// RBFFeatures
			case 0: testRBFFeatures(xDim, xMin, xMax, xSamples); break;
			// FourierRandomFeatures
			case 1: testFourierRandomFeatures(xDim, xSamples); break;
		}
		
		Utils.waitForKeypress();
	}

	private static void testFourierRandomFeatures(int xDim, double[] xSamples)
			throws IOException {
		final int nbFRF = 11;
		final double sigmaFRF = 1.;
		final @NonNull FourierRandomFeatures frf =
				new FourierRandomFeatures(xDim, nbFRF, sigmaFRF);
		plotFeatures(frf, xSamples);
	}

	private static void testRBFFeatures(int xDim,
										double[] xMin,
										double[] xMax,
										double[] xSamples)
									   throws IOException {
		final int nbRBFFeatPerDim = 11;
		final double sigmaRBF = 0.1;
		final boolean normalizedRBF = true;
		@NonNull final double[][] stateCenters =
				ArrUtils.buildGrid(xMin,xMax,nbRBFFeatPerDim);
		@NonNull final RBFFeatures rbf =
				new RBFFeatures(stateCenters,
								ArrUtils.constvec(xDim,sigmaRBF),
								normalizedRBF);
		plotFeatures(rbf, xSamples);
	}

	@SuppressWarnings("null")
	private static void plotFeatures(Features feat,
									 double[] xSamples)
									throws IOException {
		final int nbRBFFeat = feat.outDim;
		final int nbSamples = xSamples.length;
		
		@NonNull final double[][] yRBF = new double[nbSamples][nbRBFFeat];
		final long startTime = System.currentTimeMillis();
		for (int i = 0; i < nbSamples; i++) {
			@NonNull final double[] x = new double[]{xSamples[i]};
			feat.phi(x, yRBF[i]);
		}
		final long endTime = System.currentTimeMillis();
		System.out.println("Time elapsed : "+(endTime-startTime));
		System.out.println(ArrUtils.toString(yRBF));
		@NonNull final Plot2D plotRBF = new Plot2D(
				feat.toString(), "x", "y");
		plotRBF.plot(xSamples,
				     ArrUtils.transpose(yRBF, nbSamples, nbRBFFeat)[0]);
		
	}

}
