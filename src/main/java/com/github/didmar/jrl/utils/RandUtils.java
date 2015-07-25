package com.github.didmar.jrl.utils;

import java.util.Random;

import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Various useful static methods from manipulating random numbers,
 * some of which are borrowed from the JavaXCSF library.
 * In many functions no safety checks are done to improve performance,
 * unless assertions are enabled (run java with the <tt>-enableassertions</tt>
 * or <tt>-ea</tt> switch).
 * 
 * @author Didier Marin
 */
public final class RandUtils {

	/** One common random number generator */
	static final Random rnd = new Random();

	/**
	 * Sets a random seed for the random number generator.
	 * 
	 * @param s
	 *            the seed to set.
	 */
	public static void setSeed(long s) {
		rnd.setSeed(s);
	}

	/**
	 * Returns the next pseudorandom, uniformly distributed {@code boolean}
	 * value from this random number generator's sequence. The values
	 * {@code true} and {@code false} are produced with (approximately) equal
	 * probability.
	 * 
	 * @return the next pseudorandom, uniformly distributed {@code boolean}
	 *         value from this random number generator's sequence
	 */
	public static boolean nextBoolean() {
		return rnd.nextBoolean();
	}

	/**
	 * Returns a pseudorandom, uniformly distributed {@code int} value between 0
	 * (inclusive) and the specified value (exclusive), drawn from this random
	 * number generator's sequence.
	 * 
	 * @param n
	 *            the bound on the random number to be returned. Must be
	 *            positive.
	 * @return the next pseudorandom, uniformly distributed {@code int} value
	 *         between {@code 0} (inclusive) and {@code n} (exclusive) from this
	 *         random number generator's sequence
	 */
	public static int nextInt(int n) {
		return rnd.nextInt(n);
	}

	/**
	 * Returns the next pseudorandom, uniformly distributed {@code double} value
	 * between {@code 0.0} (inclusive) and {@code 1.0} (exclusive) from the
	 * random number generator's sequence.
	 * 
	 * @return the next pseudorandom, uniformly distributed {@code double} value
	 *         between {@code 0.0} and {@code 1.0} from this random number
	 *         generator's sequence
	 */
	public static double nextDouble() {
		return rnd.nextDouble();
	}

	/**
	 * Returns the next pseudorandom, Gaussian ("normally") distributed
	 * {@code double} value with mean {@code 0.0} and standard deviation
	 * {@code sigma} from the random number generator's sequence. In other
	 * words, the returned value is taken from the distribution
	 * 
	 * <pre>
	 * N(0,sigma²)
	 * </pre>
	 * 
	 * @param sigma
	 *            the standard deviation of the normal distribution
	 * @return the next pseudorandom, Gaussian ("normally") distributed
	 *         {@code double} value with mean {@code 0.0} and standard deviation
	 *         as given by {@code sigma} from this random number generator's
	 *         sequence
	 */
	public static double nextGaussian(double sigma) {
		return sigma == 0 ? 0 : rnd.nextGaussian() * sigma;
	}

	/**
	 * Drawn from the normal distribution of mean mu and vector sigma as the
	 * diagonal of the covariance matrix, and store the result in a given
	 * vector.
	 * 
	 * @param mu
	 *            the mean of the normal distribution
	 * @param sigma
	 *            the standard deviation of the normal distribution
	 * @param vec 
	 *            the vector to store the sample from the normal distribution
	 *            N(mu,sigma²)
	 * @see nextGaussian
	 */
	public static void normal(double[] mu, double[] sigma, double[] vec) {
		assert mu != null;
		assert sigma != null;
		assert vec != null;
		assert mu.length == sigma.length;
		assert vec.length >= sigma.length;
		
		for(int i=0; i<vec.length; i++) {
			vec[i] = mu[i] + nextGaussian(sigma[i]);
		}
	}

	/**
	 * Returns a vector drawn from the normal distribution of mean mu
	 * and vector sigma as the diagonal of the covariance matrix.
	 * 
	 * @param mu
	 *            the mean of the normal distribution
	 * @param sigma
	 *            the standard deviation of the normal distribution
	 * @return a vector drawn from the normal distribution N(mu,sigma²)
	 * @see Random#nextGaussian()
	 */
	public static double[] normal(double[] mu, double[] sigma) {
		assert mu != null;
		assert sigma != null;
		assert mu.length == sigma.length;
		
		final double[] vec = mu.clone();
		for(int i=0; i<vec.length; i++) {
			vec[i] += nextGaussian(sigma[i]);
		}
		return vec;
	}

	/**
	 * Returns a vector drawn from the normal distribution of zero mean and
	 * unit variance, given the length of this vector.
	 * 
	 * @param length
	 *            length of the random vector.
	 * @return a vector drawn from the normal distribution N(0,1)
	 * @see Random#nextGaussian()
	 */
	public static double[] normal(int length) {
		assert length > 0;
		
		double[] vec = new double[length];
		for(int i=0; i<length; i++) {
			vec[i] = nextGaussian(1.);
		}
		return vec;
	}

	/**
	 * Draw from a discrete probability distribution.
	 * Precondition : prob sums to 1.
	 * @param prob    a table of probabilities
	 * @return the index of the drawn event in the probability table.
	 */
	public static int drawFromDiscreteProbTable(double[] prob) {
		assert prob != null;
		assert Utils.allClose(ArrUtils.sum(prob),1.,Utils.getMacheps());
		
		final double[] cumprob = ArrUtils.cumSum(prob);
	    final double r = nextDouble();
	    // Go through prob cumulative sum
	    int i=0;
	    for(; i<prob.length; i++) {
	        if(r < cumprob[i]) {
	        	break;
	        }
	    }
	    return i;
	}

	/**
	 * Return a random permutation of the array [0,1,2,...,n-1].
	 * @param n length of the array to permute
	 * @return a random permutation of the array [0,1,2,...,n-1]
	 */
	public static int[] randPerm(int n) {
		assert n > 0;
		
		final double[] array = new double[n];
		for(int i=0; i<n; i++) {
			array[i] = nextDouble();
		}
		final int[] index = new int[n];
		Utils.quicksort(array, index);
		return index;
	}
	
}
