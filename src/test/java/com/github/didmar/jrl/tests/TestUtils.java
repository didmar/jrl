package com.github.didmar.jrl.tests;

import static org.junit.Assert.*;

import org.eclipse.jdt.annotation.NonNull;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

public class TestUtils {

	static final int SIZE_BIGVEC = 100;
	private final double[] bigvec = new double[SIZE_BIGVEC];
	private final int[] index = new int[SIZE_BIGVEC];
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		for(int i=0; i<bigvec.length; i++) {
			bigvec[i] = RandUtils.nextDouble();
		}
		
	}

	/**
	 * @throws java.lang.Exception
	 */
	@After
	public void tearDown() throws Exception {
		// Nothing to do
	}
	
	/**
	 * Test method for {@link com.github.didmar.jrl.utils.Utils#quicksort(double[], int[])}.
	 * @throws Exception 
	 */
	@Test
	public void testQuicksortDoubleArrayIntArray() throws Exception {
		@NonNull final double[] sortedBigvec = ArrUtils.cloneVec(this.bigvec);
		Utils.quicksort(sortedBigvec, this.index);
		assertTrue(this.bigvec.length == sortedBigvec.length);
		assertTrue(this.index.length == sortedBigvec.length);
		for(int i=0; i<sortedBigvec.length; i++) {
			if(i < sortedBigvec.length-1) {
				assertTrue(sortedBigvec[i] <= sortedBigvec[i+1]);				
			}
			assertTrue(sortedBigvec[i] == this.bigvec[this.index[i]]);
		}
	}

}
