package com.github.didmar.jrl.evaluation.valuefunction;

import com.github.didmar.jrl.utils.ParametricFunction;

/**
 * State-action value function based on a real vector of parameters
 * @author Didier Marin
 */
public abstract class ParametricQFunction extends QFunction implements
	ParametricFunction {
	
	/** Parameters vector */
	protected final double[] w;
	/** Length of the parameters vector */
	protected final int n;
	
	public ParametricQFunction(double[] w) {
		// TODO check w != null
		this.w = w;
		n = getParamsSize();
	}
	
	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#getParams()
	 */
	public final double[] getParams() {
		return w;
	}
	
	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#getParamsSize()
	 */
	public final int getParamsSize() {
		return w.length;
	}
	
	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#setParams(double[])
	 */
	public final void setParams(double[] w) {
		if(this.w.length != w.length) {
			throw new IllegalArgumentException("Invalid parameter size");
		}
		System.arraycopy(w,0,this.w,0,n);
	}
	
	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#updateParams(double[])
	 */
	public final void updateParams(double[] delta) {
		if(delta == null) {
			throw new IllegalArgumentException("Invalid parameter : null");
		}
		final int l = w.length;
		if(l != delta.length) {
			throw new IllegalArgumentException("Invalid parameter size");
		}
		for(int i=0; i<l; i++) {
			w[i] += delta[i];
		}
		boundParams(w);
	}
}
