package com.github.didmar.jrl.evaluation.valuefunction;

import com.github.didmar.jrl.utils.ParametricFunction;

/**
 * State value function based on a real vector of parameters 
 * @author Didier Marin
 */
public abstract class ParametricVFunction extends VFunction
	implements ParametricFunction {

	/** Parameters vector */
	protected final double[] v;
	/** Length of the parameters vector */
	protected final int n;
	
	public ParametricVFunction(double[] v) {
		if(v == null) {
			throw new IllegalArgumentException("Invalid parameter : null");
		}
		this.v = v;
		n = this.v.length;
	}
	
	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#getParams()
	 */
	public final double[] getParams() {
		return v;
	}
	
	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#getParamsSize()
	 */
	public final int getParamsSize() {
		return n;
	}
	
	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#setParams(double[])
	 */
	public final void setParams(double[] v) {
		if(this.v.length != v.length) {
			throw new IllegalArgumentException("Invalid parameter size");
		}
		System.arraycopy(v,0,this.v,0,n);
	}

	/* (non-Javadoc)
	 * @see jrl.utils.ParametricFunction#updateParams(double[])
	 */
	public final void updateParams(double[] delta) {
		if(delta == null) {
			throw new IllegalArgumentException("Invalid parameter : null");
		}
		final int l = v.length;
		if(l != delta.length) {
			throw new IllegalArgumentException("Invalid parameter size");
		}
		for(int i=0; i<l; i++) {
			v[i] += delta[i];
		}
		boundParams(v);
	}

}
