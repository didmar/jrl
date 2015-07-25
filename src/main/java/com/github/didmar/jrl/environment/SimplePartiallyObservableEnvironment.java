package com.github.didmar.jrl.environment;

import com.github.didmar.jrl.environment.IEnvironment;

/**
 * Simple implementation of the {@link PartiallyObservableEnvironment} interface
 * which compute the observation as a selection of some of the state dimensions. 
 * @author Didier Marin
 */
public final class SimplePartiallyObservableEnvironment extends
		PartiallyObservableEnvironment {
	
	/** The state-space dimension to be put in the observation */
	private final int[] selection;
	
	public SimplePartiallyObservableEnvironment(IEnvironment fullyObsEnv,
			int[] selection) {
		super(fullyObsEnv, selection.length);
		if(fullyObsEnv == null) {
			throw new IllegalArgumentException("fullyObsEnv must be non-null");
		}
		this.selection = selection;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.PartiallyObservableEnvironment#computeObservation(double[], double[])
	 */
	@Override
	public final void computeObservation(double[] x, double[] o) {
		assert x != null;
		assert x.length == getXDim();
		assert o.length == getODim();
		
		for(int i=0; i<getODim(); i++) {
			o[i] = x[selection[i]];
		}
	}

}
