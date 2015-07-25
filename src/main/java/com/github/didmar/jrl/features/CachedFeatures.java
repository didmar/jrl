package com.github.didmar.jrl.features;

import java.util.HashMap;
import java.util.Map;

import org.eclipse.jdt.annotation.NonNull;
import org.eclipse.jdt.annotation.Nullable;

import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Decorator to cache some features, which might speed things up
 * when the same inputs are used very often.
 * 
 * @author Didier Marin
 */
public final class CachedFeatures extends Features {

	/** The features to cache */
	private final Features baseFeat;
	/** The cache, which maps inputs to their corresponding output */ 
	private final Map<double[],double[]> cache;
	
	public CachedFeatures(final Features baseFeat, int initialCapacity) {
		super(baseFeat.inDim, baseFeat.outDim);
		this.baseFeat = baseFeat;
		cache = new HashMap<double[],double[]>(initialCapacity);
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#isNormalized()
	 */
	@Override
	public final boolean isNormalized() {
		return baseFeat.isNormalized();
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#phi(double[], double[])
	 */
	@Override
	public final void phi(@NonNull final double[] x, @NonNull final double[] y)
			throws IllegalArgumentException {
		assert x != null;
		assert y != null;
		
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		if(y.length != outDim ){
			throw new IllegalArgumentException("y must have length outDim");
		}
		
		@Nullable final double[] cachedY = cache.get(x);
		if(cachedY == null) {
			baseFeat.phi(x, y);
			cache.put(x, y);
		} else {
			System.arraycopy(cachedY, 0, y, 0, outDim);
		}
		
		assert (!isNormalized()) || Utils.allClose(ArrUtils.sum(y),1.,Utils.getMacheps())
			: "Features should be normalized";
	}

}
