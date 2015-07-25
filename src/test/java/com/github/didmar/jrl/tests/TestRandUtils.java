package com.github.didmar.jrl.tests;

import static org.junit.Assert.*;

import org.junit.Test;

import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

public class TestRandUtils {

	/**
	 * Test method for {@link com.github.didmar.jrl.utils.RandUtils#normal(double[], double[])}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testNormal() {
		final double[] samples = new double[10000];
		final double[] mu = {0.};
		final double[] sigma = {1.};
		final double[] tmpvec = new double[1];
		for(int i=0; i<samples.length; i++) {
			RandUtils.normal(mu, sigma, tmpvec);
			samples[i] = tmpvec[0];
		}
		assertTrue(Math.abs( ArrUtils.mean(samples) - mu[0]) < 0.1);
		assertTrue(Math.abs( ArrUtils.std(samples) - sigma[0]) < 0.1);
	}
}
