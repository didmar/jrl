package com.github.didmar.jrl.utils;

//TODO find a better name for this : it is also used for lambdas and kappas... 
/**
 * A discount factor is a floating number between 0 and 1.
 * 
 * @author Didier Marin
 */
public class DiscountFactor {
	
	public final double value;
	
	public DiscountFactor(double value) {
		if(value < 0. || value > 1.) {
			throw new IllegalArgumentException("Discount factor must be in [0,1]");
		}
		this.value = value;
	}

	public final double mixture(double a, double b) {
		return value * a + (1 - value) * b;
	}
}