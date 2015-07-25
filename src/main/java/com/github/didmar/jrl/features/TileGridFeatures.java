package com.github.didmar.jrl.features;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Binary tile coding features : divide the input space into a grid of
 * cells, which output 1 if the input 
 * 
 * @author Didier Marin
 */
public final class TileGridFeatures extends Features {

	private final double[] mins;
	private final double[] maxs;
	private final int[] steps;

	public TileGridFeatures(final double[] mins,
							final double[] maxs,
							int step)
						   throws IllegalArgumentException {
		super(mins.length, ((int)Math.pow(step,mins.length))+1);
		if(mins.length != maxs.length) {
			throw new IllegalArgumentException("mins and maxs must have the same size");
		}
		if(step <= 0) {
			throw new IllegalArgumentException("step must be greater than 0");
		}
		this.mins = mins;
		this.maxs = maxs;
		this.steps = new int[inDim];
		for(int i=0; i<inDim; i++) {
			steps[i] = step;
		}
	}
	
	public TileGridFeatures(final double[] mins,
							final double[] maxs,
							final int[] steps)
						   throws IllegalArgumentException {
		super(steps.length, getNbTiles(steps)+1);
		if(mins.length != maxs.length) {
			throw new IllegalArgumentException("mins and maxs must have the same size");
		}
		for(int step : steps) {
			if(step <= 0) {
				throw new IllegalArgumentException("step must be greater than 0");
			}
		}
		this.mins = mins;
		this.maxs = maxs;
		this.steps = steps;
	}
	
	private static final int getNbTiles(int[] steps) {
		int nbTiles = 1;
		for(int i=0; i<steps.length; i++) {
			nbTiles *= steps[i];
		}
		return nbTiles;
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#isNormalized()
	 */
	@Override
	public final boolean isNormalized() {
		return true;
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#phi(double[], double[])
	 */
	@Override
	public final void phi(@NonNull final double[] x, @NonNull final double[] y) {
		assert x != null;
		assert y != null;
		
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		if(y.length != outDim ){
			throw new IllegalArgumentException("y must have length outDim");
		}
		
		ArrUtils.zeros(y);
		int k = 1;
		int ind = 0;
		for(int i=0; i<steps.length; i++) {
			final double relNormX = (x[i]-mins[i])/(maxs[i]-mins[i]);
			// Outside of the grid ?
			if(relNormX < 0. || relNormX > 1.) {
				y[outDim-1] = 1.;
				return;
			}
			// Find which tile it matches
			ind += k*Math.floor(relNormX*(steps[i]-1));
			k *= steps[i];
		}
		y[ind] = 1.;
		
		assert Utils.allClose(ArrUtils.sum(y),1.,Utils.getMacheps()) : "Features are not normalized";
	}

}
