package com.github.didmar.jrl.utils.array;

import java.util.Arrays;

import org.eclipse.jdt.annotation.Nullable;

import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.Utils;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

/**
 * Various useful static methods from manipulating multi-dimensional arrays,
 * some of which are borrowed from the JavaXCSF library.
 * Some functions depend on the JAMA library v1.0.3
 * 
 * In many functions no safety checks are done to improve performance,
 * unless assertions are enabled (run java with the <tt>-enableassertions</tt>
 * or <tt>-ea</tt> switch).
 * 
 * @author Didier Marin
 */
public final class ArrUtils {

	/**
	 * Convergence indicator for
	 * {@link tql2}, which is used in
	 * eigendecompositions.
	 */
	static final double EPSILON = Math.pow(2.0, -52.0);
	
	/**
	 * Returns a clone of a vector, guaranting a non null result 
	 * @param vec	a vector
	 * @return a clone of vec
	 */
	public static final double[] cloneVec(final double[] vec) {
		@Nullable final double[] clone = vec.clone();
		if(clone == null) throw new RuntimeException("Could not clone vector");
		return clone;
	}
	
	/**
	* Computes the sum of the elements of a vector.
	* @param array  the array, to be summed.
	* @return       the sum of the elements of the vector.
	*/
	public static double sum(final double[] vec) {
		assert vec != null;
		
		double s = 0;
		for(double i : vec) {
			s += i;
		}
		return s;
	}

	/**
	 * Compute a vector res such that for all i in [0,vec.length[:
	 * 		res[i] = vec[0] + vec[1] + ... + vec[i]
	 * @param vec	the vector vec
	 * @return	the vector res
	 */
	public static double[] cumSum(final double[] vec) {
		assert vec != null;
		
		final int l = vec.length;
		final double[] cumSummedArray = new double[l];
		double sum = 0.;
		for(int i=0; i<l; i++) {
			cumSummedArray[i] = vec[i] + sum;
			sum += vec[i];
		}
		return cumSummedArray;
	}

	/**
	* Computes the mean of the elements of a vector.
	* @param vec	the vector.
	* @return       the mean of the elements of the vector.
	*/
	public static double mean(final double[] vec) {
		assert vec != null;
		
		return sum(vec) / vec.length;
	}

	/**
	* Computes the minimum of the elements of a vector.
	* @param vec	the vector.
	* @return		the minimum of the elements of the vector.
	*/
	public static double min(final double[] vec) {
		assert vec != null;
		
		double min = vec[0];
		for(double i : vec) {
			if(i < min){
				min = i;
			}
		}
		return min;
	}

	/**
	* Computes the maximum of the elements of a vector.
	* @param vec	the vector.
	* @return		the maximum of the elements of the vector.
	*/
	public static double max(final double[] vec) {
		assert vec != null;
		
		double max = vec[0];
		for(double i : vec) {
			if(i > max){
				max = i;
			}
		}
		return max;
	}

	/**
	 * Returns the index of the smallest element in a vector.
	 * Precondition : vec != null
	 * @param vec    the vector to find the argmin of
	 * @return the argmin of vec
	 */
	public static int argmin(final double[] vec) {
		assert vec != null;
		
		double min = vec[0];
		int argmin = 0;
		for(int i=1; i<vec.length; i++) {
			final double tmp = vec[i];
			if(tmp < min) {
				min = tmp;
				argmin = i;
			}
		}
		return argmin;
	}

	/**
	 * Returns the index of the greatest element in a vector.
	 * Precondition : vec != null
	 * @param vec    the vector to find the argmax of
	 * @return the argmin of vec
	 */
	public static int argmax(final double[] vec) {
		assert(vec != null);
		double max = vec[0];
		int argmax = 0;
		for(int i=1; i<vec.length; i++) {
			final double tmp = vec[i];
			if(tmp > max) {
				max = tmp;
				argmax = i;
			}
		}
		return argmax;
	}

	/**
	 * Computes the sample variance of the given values, that is the quadratic
	 * deviation from mean.
	 * @param vec	the vector to compute the variance of
	 * @return		the variance of <code>vec</code>
	 */
	public static double var(final double[] vec) {
		assert vec != null;
		
		if(vec.length == 1) {
			return 0.;
		}
		final double mean = mean(vec);
		double var = 0;
		for (double d : vec) {
			var += (d - mean) * (d - mean);
		}
		var /= (vec.length - 1);
		return var;
	}

	/**
	 * Computes the standard deviation of the given values, that is the square
	 * root of the sample variance.
	 * @param vec	the vector to compute the standard deviation of
	 * @return		the standard deviation of <code>vec</code>
	 */
	public static double std(final double[] vec) {
		assert vec != null;
		
		return Math.sqrt(var(vec));
	}

	/**
	* Reallocates an array with a new size, and copies the contents
	* of the old array to the new array.
	* @param oldArray  the old array, to be reallocated.
	* @param newSize   the new array size.
	* @return          a new array with the same contents.
	*/
	@SuppressWarnings("null")
	public static Object resizeArray(final Object oldArray, int newSize) {
		assert oldArray != null;
		assert newSize > 0;
		
		final int oldSize = java.lang.reflect.Array.getLength(oldArray);
		Class<?> elementType = oldArray.getClass().getComponentType();
		final Object newArray = java.lang.reflect.Array.newInstance(elementType,newSize);
		final int preserveLength = Math.min(oldSize,newSize);
		if (preserveLength > 0)
			System.arraycopy (oldArray,0,newArray,0,preserveLength);
		return newArray;
	}

	/**
	 * Returns whether matrix mat has size l1-by-l2 or not 
	 * @param mat	a matrix
	 * @param l1	first dimension (lines)
	 * @param l2	second dimension (columns)
	 * @return
	 */
	public static boolean hasShape(final double[][] mat, int l1, int l2) {
		assert mat != null;
		assert l1 > 0;
		assert l2 > 0;
		
		if(mat.length != l1) {
			return false;
		}
		for (int i = 0; i < mat.length; i++) {
			if(mat[i].length != l2) {
				return false;
			}
		}
		return true;
	}

	public static boolean hasShape(final double[][][] mat3d,
			int l1, int l2, int l3) {
		assert mat3d != null;
		assert l1 > 0;
		assert l2 > 0;
		assert l3 > 0;
		
		if(mat3d.length != l1) {
			return false;
		}
		for (int i = 0; i < mat3d.length; i++) {
			if(mat3d[i].length != l2) {
				return false;
			}
			for (int j = 0; j < mat3d[i].length; j++) {
				if(mat3d[i][j].length != l3) {
					return false;
				}
			}
		}
		return true;
	}

	/**
	 * Faster implementation of the {@link Arrays#equals(Object)} method without
	 * length comparison for two double arrays. Note that this method does no
	 * length checks for performance reasons. The behavior is undefined, if
	 * 
	 * <pre>
	 * array1.length != array2.length
	 * </pre>
	 * 
	 * @param array1
	 *            the first double array
	 * @param array2
	 *            the second double array of the same length (this is not
	 *            checked!)
	 * @return <code>true</code>, if the arrays contain the same values;
	 *         <code>false</code> otherwise.
	 */
	public static boolean arrayEquals(@Nullable final double[] array1,
									  @Nullable final double[] array2) {
		if (array1 == array2) {
			return true;
		} else if (array1 == null || array2 == null) {
			return false;
		}
		for (int i = 0; i < array1.length; i++) {
			if (array1[i] != array2[i]) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Faster implementation of the {@link Arrays#equals(Object)} method without
	 * length comparison for two double arrays. Note that this method does no
	 * length checks for performance reasons.
	 * 
	 * @param array1
	 *            the first double array
	 * @param array2
	 *            the second double array of the same size (this is not
	 *            checked!)
	 * @return <code>true</code>, if the arrays contain the same values;
	 *         <code>false</code> otherwise.
	 */
	public static boolean arrayEquals(@Nullable final double[][] array1,
									  @Nullable final double[][] array2) {
		if (array1 == null || array2 == null) {
			return false;
		}
		assert array1.length == array2.length;
		for (int i = 0; i < array1.length; i++) {
			assert array1[i].length == array2[i].length;
			for (int j = 0; j < array1.length; j++) {
				if (array1[i][j] != array2[i][j]) {
					return false;
				}
			}
		}
		return true;
	}

	/**
	 * Returns <code>true</code> if two arrays are element-wise equal within a
	 * tolerance.
	 * 
	 * @param array1
	 *            the first double array
	 * @param array2
	 *            the second double array of the same length (this is not
	 *            checked!)
	 * @param tol
	 * 			  tolerance parameter
	 * @return <code>true</code>, if the arrays contain the same values within
	 * 			tolerance <code>tol</code>; <code>false</code> otherwise.
	 */
	public static boolean allClose(final double[][] array1,
								   final double[][] array2,
								   double tol) {
		assert array1.length == array2.length;
		assert tol >= 0.;
		
		for (int i = 0; i < array1.length; i++) {
			assert array1[i].length == array2[i].length;
			for (int j = 0; j < array1.length; j++) {
				if(!Utils.allClose(array1[i][j], array2[i][j], tol)) {
					return false;
				}
			}
		}
		return true;
	}

	public static boolean allGreaterOrEqual(final double[] x,
											final double[] y) {
		assert x != null;
		assert y != null;
		assert x.length <= y.length;
		
		for (int i=0; i<x.length; i++) {
			if(x[i] < y[i]) {
				return false;
			}
		}
		return true;
	}

	public static boolean allLessOrEqual(final double[] x,
										 final double[] y) {
		assert x != null;
		assert y != null;
		assert x.length <= y.length;
		
		for (int i=0; i<x.length; i++) {
			if(x[i] > y[i]) {
				return false;
			}
		}
		return true;
	}

	public static boolean allGreater(final double[] x,
									 final double[] y) {
		assert x != null;
		assert y != null;
		assert x.length <= y.length;
		
		for (int i=0; i<x.length; i++) {
			if(x[i] <= y[i]) {
				return false;
			}
		}
		return true;
	}

	public static boolean allLess(final double[] x, final double[] y) {
		assert x != null;
		assert y != null;
		assert x.length <= y.length;
		
		for (int i=0; i<x.length; i++) {
			if(x[i] >= y[i]) {
				return false;
			}
		}
		return true;
	}

	/**
	* Returns a vector of length l filled with the constant cst
	* @param l    the length of the vector, must be greater than 0.
	* @param cst  the constant we will fill the vector with.
	* @return     a vector of length l filled with the constant cst.
	*/
	public static double[] constvec(int l, double cst) {
		assert l > 0;
		final double[] vec = new double[l];
		ArrUtils.constvec(vec,cst);
		return vec;
	}

	/**
	* Returns a vector of length l filled with ones.
	* @param l    the length of the vector.
	* @return     a vector of length l filled with ones.
	*/
	public static double[] ones(int l) {
		assert l > 0;
		return constvec(l,1.);
	}

	/**
	* Returns a vector of length l filled with zeros.
	* @param l    the length of the vector.
	* @return     a vector of length l filled with zeros.
	*/
	public static double[] zeros(int l) {
		assert l > 0;
		return constvec(l,0.);
	}

	/**
	* Returns a vector of length l filled with random values between 0 and 1.
	* @param l    the length of the vector.
	* @return     a vector of length l filled with random values.
	*/
	public static double[] rand(int l) {
		assert l > 0;
		final double[] vec = new double[l];
		ArrUtils.rand(vec);
		return vec;
	}

	/**
	* Fills a vector with the constant cst.
	* Precondition : vec != null
	* @param vec  the vector to fill.
	* @param cst  the constant we will fill the vector with.
	* 
	*/
	public static void constvec(final double[] vec, double cst) {
		assert vec != null;
		for(int i=0; i<vec.length; i++) {
			vec[i] = cst;
		}
	}

	/**
	* Fills a vector with ones.
	* @param vec  the vector to fill with ones.
	* 
	*/
	public static void ones(final double[] vec) {
		assert vec != null;
		constvec(vec,1.);
	}

	/**
	* Fills a vector with zeros.
	* @param vec  the vector to fill with zeros.
	* 
	*/
	public static void zeros(final double[] vec) {
		assert vec != null;
		constvec(vec,0.);
	}

	/**
	* Fills a vector with random values between 0 and 1.
	* @param vec  the vector to fill with random values.
	* 
	*/
	public static void rand(final double[] vec) {
		assert vec != null;
		for(int i=0; i<vec.length; i++) {
			vec[i] = RandUtils.nextDouble();
		}
	}

	/**
	* Returns a l1 x l2 matrix without filling anything.
	* @param l1    number of lines of the matrix
	* @param l2    number of columns of the matrix
	* @return      l1 x l2 matrix.
	*/
	public static double[][] empty(int l1, int l2) {
		assert l1 > 0;
		assert l2 > 0;
		return new double[l1][l2];
	}

	/**
	* Returns a new unfilled matrix with the same size as another matrix.
	* @param A    a matrix from which to get the size
	* @return     a new unfilled matrix of the same size as A.
	*/
	public static double[][] emptyLike(final double[][] A) {
		assert A != null;
		final double[][] mat = new double[A.length][];
		for (int i = 0; i < mat.length; i++) {
			assert A[i] != null;
			mat[i] = new double[A[i].length];
		}
		return mat;
	}

	/**
	* Returns a l1 x l2 matrix filled with a constant.
	* @param l1    number of lines of the matrix
	* @param l2    number of columns of the matrix
	* @param cst   the constant to fill the matrix with
	* @return      l1 x l2 matrix filled with constant cst.
	*/
	public static double[][] constmat(int l1, int l2, double cst) {
		assert l1 > 0;
		assert l2 > 0;
		final double[][] mat = empty(l1,l2);
		ArrUtils.constmat(mat,cst);
		return mat;
	}

	/**
	* Returns a l1 x l2 matrix filled with ones.
	* @param l1    number of lines of the matrix
	* @param l2    number of columns of the matrix
	* @return      l1 x l2 matrix filled with ones.
	*/
	public static double[][] ones(int l1, int l2) {
		assert l1 > 0;
		assert l2 > 0;
		return constmat(l1,l2,1.);
	}

	/**
	* Returns a l1 x l2 matrix filled with zeros.
	* @param l1    number of lines of the matrix
	* @param l2    number of columns of the matrix
	* @return      l1 x l2 matrix filled with zeros.
	*/
	public static double[][] zeros(int l1, int l2) {
		assert l1 > 0;
		assert l2 > 0;
		return constmat(l1,l2,0.);
	}

	/**
	* Fills a matrix with the constant cst
	* @param mat  the matrix to fill.
	* @param cst  the constant we will fill the matrix with.
	* 
	*/
	public static void constmat(final double[][] mat, double cst) {
		assert mat != null;
		for(int i=0; i<mat.length; i++) {
			assert mat[i] != null;
			for(int j=0; j<mat[i].length; j++) {
				mat[i][j] = cst;
			}
		}
	}

	/**
	* Fills a matrix with the constant cst
	* @param mat  the matrix to fill.
	* @param cst  the constant we will fill the matrix with.
	* 
	*/
	public static void constmat(final int[][] mat, int cst) {
		assert mat != null;
		for(int i=0; i<mat.length; i++) {
			assert mat[i] != null;
			for(int j=0; j<mat[i].length; j++) {
				mat[i][j] = cst;
			}
		}
	}

	/**
	* Fills a matrix with ones.
	* @param mat  the matrix to fill with ones.
	* 
	*/
	public static void ones(final double[][] mat) {
		assert mat != null;
		constmat(mat,1.);
	}

	/**
	* Fills a matrix with zeros.
	* @param mat  the matrix to fill with zeros.
	*/
	public static void zeros(final double[][] mat) {
		assert(mat != null);
		constmat(mat,0.);
	}

	/**
	* Returns an n x n identity matrix.
	* @param n dimension of the square matrix
	*/
	public static double[][] eye(int n) {
		assert n > 0;
		
		final double[][] mat = zeros(n,n);
		for(int i=0; i<n; i++) {
			mat[i][i] = 1.;
		}
		return mat;
	}

	public static double[][] eye(int n, double diag) {
		assert n > 0;
		
		final double[][] mat = zeros(n,n);
		ArrUtils.eye(mat, diag);
		return mat;
	}

	public static void eye(final double[][] mat, double diag) {
		assert isQuadratic(mat);
		
		if(diag != 0.) {
			for(int i=0; i<mat.length; i++) {
				mat[i][i] = diag;
			}
		}
	}

	/**
	* Filled a matrix with random values between 0 and 1.
	* @param mat	the matrix to fill
	*/
	public static void rand(final double[][] mat) {
		assert mat != null;
		
		for(int i=0; i<mat.length; i++) {
			for(int j=0; j<mat[i].length; j++) {
				mat[i][j] = RandUtils.nextDouble();
			}
		}
	}

	/**
	* Returns a l1-by-l2 matrix filled with random values between 0 and 1.
	* @param l1   number of rows
	* @param l1   number of columns
	* @return     a l1-by-l2 matrix filled with random values between 0 and 1.
	*/
	public static double[][] rand(int l1, int l2) {
		assert l1 > 0;
		assert l2 > 0;
		
		final double[][] mat = new double[l1][l2];
		rand(mat);
		return mat;
	}

	/**
	* Returns the euclidean norm of a vector.
	* @param vec  the vector.
	* @return     the euclidean norm of vector vec.
	* 
	*/
	public static double norm(final double[] vec) {
		assert vec != null;
		
		double s = 0;
		for(double i : vec) {
			s += Math.pow(i,2);
		}
		return Math.sqrt(s);
	}

	/**
	* Returns the square of the euclidean norm of a vector.
	* @param vec  the vector.
	* @return     the square of the euclidean norm of vector vec.
	* 
	*/
	public static double squaredNorm(final double[] vec) {
		assert vec != null;
		
		double s = 0;
		for(double i : vec) {
			s += Math.pow(i,2);
		}
		return s;
	}

	/**
	* Bound a vector
	* @param vec     the vector to be bounded.
	* @param bounds  a matrix that gives for each dimension (lines) a minimum
	*                and maximum value (first and second column, resp.).
	* @return whether some components were bounded or not
	*/
	public static boolean boundVector(final double[] vec,
								   	  final double[][] bounds) {
		assert vec != null;
		assert bounds != null;
		assert vec.length == bounds.length;
		
		boolean bounded = false;
		for(int i=0; i<vec.length; i++) {
			assert bounds[i].length == 2;
			if(vec[i] < bounds[i][0]) {
				vec[i] = bounds[i][0];
				bounded = true;
			} else if(vec[i] > bounds[i][1]) {
				vec[i] = bounds[i][1];
				bounded = true;
			}
		}
		return bounded;
	}

	/**
	* Bound a vector. min or max may be null, which means we don't need to
	* check the lower or upper bound, respectively.
	* @param vec	the vector to be bounded.
	* @param min	a vector that gives the minimum value for each dimension
	* @param max	a vector that gives the maximum value for each dimension
	* @return whether some components were bounded or not
	*/
	public static boolean boundVector(final double[] vec,
								   final double[] min,
								   final double[] max) {
		assert vec.length == min.length;
		assert vec.length == max.length;

		// Build Nx2 bounds array where bounds[i] = {min[i], max[i]}
		double[][] bounds = new double[vec.length][2];
		for (int i = 0; i < vec.length; i++) {
			bounds[i][0] = min[i];
			bounds[i][1] = max[i];
		}
		return boundVector(vec, bounds);
	}

	/**
	* Bound each component of a vector between min or max.
	* @param vec	the vector to be bounded.
	* @param min	the minimum value for any dimension
	* @param max	the maximum value for any dimension
	* @return whether some components were bounded or not
	*/
	public static boolean boundVector(final double[] vec,
								   	  double min, double max) {
		assert min < max;
		// Build Nx2 bounds array where bounds[i] = {min, max}
		double[][] bounds = new double[vec.length][2];
		for (int i = 0; i < vec.length; i++) {
			bounds[i][0] = min;
			bounds[i][1] = max;
		}
		return boundVector(vec, bounds);
	}
		
	
	/**
	 * Normalize a vector, i.e. divide each component by the sum of all the
	 * components. If the sum is 0, the vector is not modified and the method
	 * returns false.
	 * Precondition : vec != null
	 * @param vec    the vector to normalize
	 * @return false if the vector summed to 0, true else
	 */
	public static boolean normalize(final double[] vec) {
		assert vec != null;
		
		final double sum = sum(vec);
		if(sum != 0.) {
			for(int i=0; i<vec.length; i++) {
				vec[i] /= sum;
			}
			assert sum(vec) == 1.;
			return true;
		}
		return false;
	}

	/**
	 * Compute the euclidean distance between 2 vectors.
	 * Preconditions : vec1.length == vec2.length == n
	 * @param vec1    a real vector
	 * @param vec2    another real vector
	 * @param n       length of two vectors
	 * @return the euclidean distance between vec1 and vec2
	 */
	public static double euclideanDist(final double[] vec1,
									   final double[] vec2,
									   int n) {
		assert vec1 != null;
		assert vec2 != null;
		assert vec1.length == n;
		assert vec2.length == n;
		
		double dist = 0;
		for(int i=0; i<n; i++) {
			dist += Math.pow(vec1[i]-vec2[i],2);
		}
		return Math.sqrt(dist);
	}
	
	/**
	 * Create a vector of n values ranging from min to max
	 * @param min	value of the first element
	 * @param max	value of the last element
	 * @param n		vector length
	 * @return		a vector of n values ranging from min to max
	 */
	public static double[] linspace(double min, double max, int n) {
		assert n > 0;
		assert min <= max;
		
		final double[] vec = new double[n];
		double step = (max-min)/(n-1);
		for(int i=0; i<n; i++) {
			vec[i] = min + i*step;
		}
		return vec;
	}

	/**
	 * Build a multi-dimensional grid, i.e. if we have :
	 * mins =[0.,-1.,10.,0.]
	 * maxs =[1., 1.,15.,1.]
	 * steps=[ 3,  2,  2, 1]
	 * the resulting grid will be the following matrix :
	 * [[0.0,-1.0,10.0,0.0],
	 *  [0.0,-1.0,15.0,0.0],
	 *  [0.0,1.0,10.0,0.0],
	 *  [0.0,1.0,15.0,0.0],
	 *  [0.5,-1.0,10.0,0.0],
	 *  [0.5,-1.0,15.0,0.0],
	 *  [0.5,1.0,10.0,0.0],
	 *  [0.5,1.0,15.0,0.0],
	 *  [1.0,-1.0,10.0,0.0],
	 *  [1.0,-1.0,15.0,0.0],
	 *  [1.0,1.0,10.0,0.0],
	 *  [1.0,1.0,15.0,0.0]]
	 * 
	 * Preconditions : mins, maxs and steps must have the same length.
	 * mins[i] <= maxs[i] for all i.
	 * all elements in steps must greater than 0.  
	 * 
	 * @param mins
	 * @param maxs
	 * @param steps
	 * @return a prod(steps)-by-len(mins) matrix where each line represents
	 *         a vertice of the multi-dimensional grid
	 */
	public static double[][] buildGrid(final double[] mins,
												final double[] maxs,
												final int[] steps) {
		assert mins != null; 
		assert maxs != null;
		assert steps != null;
		assert mins.length == maxs.length; 
		assert mins.length == steps.length;
		
		final int dim = mins.length;
		int nbPts = 1; // Number of grid intersections
	    for(int i=0; i<dim; i++) {
	        assert steps[i] > 0;
	        nbPts *= steps[i];
	    }
	    // Init centers
	    final double[][] c = new double[nbPts][dim];
	    int p = 1;
	    // Assign the coordinates for each dimension
	    for(int j=0; j<dim; j++) {
	        for(int i=0; i<nbPts; i++) {
	        	// ind is in [ 0, steps[j] [
	            final int ind = (new Double(Math.floor(Math.floor((i%(nbPts/p)))/(nbPts/(p*steps[j]))))).intValue();
	            final int s = steps[j]-1;
	            c[i][j] = mins[j] + (s == 0 ? 0. : ind*(maxs[j]-mins[j])/s);
	        }
	        p *= steps[j];
	    }
	    return c;
	}

	public static double[][] buildGrid(final double[] mins,
												final double[] maxs,
												int step) {
		assert mins != null; 
		assert maxs != null;
		assert step > 0;
		assert mins.length == maxs.length; 
		
		final int[] steps = new int[mins.length];
		for(int i=0; i<steps.length; i++) {
			steps[i] = step;
		}
		return buildGrid(mins,maxs,steps);
	}
	
	/**
	 * Check that the given 2d array is a matrix
	 * (i.e. same length for all nested arrays)
	 * @param mat
	 * @return whether mat shape is that of a matrix
	 */
	public static boolean isMatrix(final double[][] mat) {
		assert mat != null;
		
		final int l1 = mat.length;
		final int l2 = mat[0].length; 
		for(int i=1; i<l1; i++) {
			assert mat[i] != null; 
			if(mat[i].length != l2) return false;
		}
		return true;
	}

	@SuppressWarnings("null")
	public static String toString(final int[][] mat) {
		assert mat != null;
		
		final StringBuilder sb = new StringBuilder("[");
		for(int i=0; i<mat.length; i++) {
			if(i>0) {
				sb.append(";\n");
			}
			assert mat[i] != null;
			sb.append(toString(mat[i]));
		}
		sb.append("]");
		return sb.toString();
	}

	@SuppressWarnings("null")
	public static String toString(final double[][] mat) {
		assert mat != null;
		
		final StringBuilder sb = new StringBuilder("[");
		for(int i=0; i<mat.length; i++) {
			if(i>0) {
				sb.append(";\n");
			}
			assert mat[i] != null;
			sb.append(toString(mat[i]));
		}
		sb.append("]");
		return sb.toString();
	}

	public static String toString(final int[] vec) {
		assert vec != null;
		
		String s = "[";
		for(int i=0; i<vec.length; i++) {
			if(i>0) {
				s += ",";
			}
			s += vec[i];
		}
		s += "]";
		return s;
	}

	public static String toString(final double[] vec) {
		assert vec != null;
		
		String s = "[";
		for(int i=0; i<vec.length; i++) {
			if(i>0) {
				s += ",";
			}
			s += vec[i];
		}
		s += "]";
		return s;
	}

	/**
	 * Multiplies the vector <tt>a</tt> (l1) with the constant <tt>b</tt>.
	 * The result is stored in <code>dest</code> (l1), that is
	 * 
	 * <pre>
	 * dest = a * b
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param a
	 *            a <tt>l1</tt> vector
	 * @param b
	 *            a real constant
	 * @param dest
	 *            [on return] <tt>l1</tt> vector that holds the
	 *            product of <tt>a</tt> and <tt>b</tt>
	 * @param n
	 *            length of <tt>a</tt>
	 */
	public static void multiply(final double[] a, double b,
								final double[] dest, int n) {
		assert a != null;
		assert dest != null;
		assert a.length >= n;
		assert dest.length >= n;
		
		for (int i = 0; i < n; i++) {
			dest[i] = a[i] * b;
		}
	}

	/**
	 * Multiplies the matrix <tt>A</tt> (l1-by-l2) with the constant <tt>b</tt>.
	 * The result is stored in <code>dest</code> (l1-by-l2), that is
	 * 
	 * <pre>
	 * dest = A * b
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param A
	 *            a <tt>l1</tt>-by-<tt>l2</tt> matrix
	 * @param b
	 *            a real constant
	 * @param dest
	 *            [on return] <tt>l1</tt>-by-<tt>l2</tt> matrix that holds the
	 *            product of <tt>A</tt> and <tt>b</tt>
	 * @param l1
	 *            number of rows of matrix A
	 * @param l2
	 *            number of columns of matrix A
	 */
	public static void multiply(final double[][] A, double b,
								final double[][] dest, int l1, int l2) {
		assert hasShape(A,l1,l2);
		assert hasShape(dest,l1,l2);
		
		for (int i = 0; i < l1; i++) {
			for (int j = 0; j < l2; j++) {
				dest[i][j] = A[i][j] * b;
			}
		}
	}

	/**
	 * Multiplies the quadratic matrix <tt>A</tt> (n-by-n) with the quadratic
	 * matrix <tt>B</tt> (n-by-n). The result is stored in <code>dest</code>
	 * (n-by-n), that is
	 * 
	 * <pre>
	 * dest = A * B
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param A
	 *            a quadratic <tt>n</tt>-by-<tt>n</tt> matrix
	 * @param B
	 *            a quadratic <tt>n</tt>-by-<tt>n</tt> matrix
	 * @param dest
	 *            this quadratic <tt>n</tt>-by-<tt>n</tt> matrix holds the
	 *            product of <tt>A</tt> and <tt>B</tt> on return
	 * @param n
	 *            size of the arrays to be multiplied
	 */
	public static void multiplyQuad(final double[][] A,
									final double[][] B,
									final double[][] dest,
									int n) {
		assert hasShape(A,n,n);
		assert hasShape(B,n,n);
		assert hasShape(dest,n,n);
		
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				dest[i][j] = A[i][0] * B[0][j];
				for (int k = 1; k < n; k++) {
					dest[i][j] += A[i][k] * B[k][j];
				}
			}
		}
	}

	/**
	 * Multiplies the matrix <tt>A</tt> (l1-by-l2) with the matrix <tt>B</tt>
	 * (l2-by-l3). The result is stored in <code>dest</code> (l1-by-l3), that is
	 * 
	 * <pre>
	 * dest = A * B
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param A
	 *            a <tt>l1</tt>-by-<tt>l2</tt> matrix
	 * @param B
	 *            a <tt>l2</tt>-by-<tt>l3</tt> matrix
	 * @param dest
	 *            this <tt>l1</tt>-by-<tt>l3</tt> matrix holds the
	 *            product of <tt>A</tt> and <tt>B</tt> on return
	 * @param l1
	 *            number of rows of matrix A
	 * @param l2
	 *            number of columns of matrix A and rows of matrix B
	 * @param l3
	 *            number of columns of matrix B
	 */
	public static void multiply(final double[][] A,
								final double[][] B,
								final double[][] dest,
								int l1, int l2, int l3) {
		assert hasShape(A,l1,l2);
		assert hasShape(B,l2,l3);
		assert hasShape(dest,l1,l3);
		
		for (int i = 0; i < l1; i++) {
			for (int j = 0; j < l3; j++) {
				dest[i][j] = A[i][0] * B[0][j];
				for (int k = 1; k < l2; k++) {
					dest[i][j] += A[i][k] * B[k][j];
				}
			}
		}
	}

	/**
	 * Multiplies the quadratic matrix <tt>A</tt> (n-by-n) with column vector
	 * <tt>v</tt> (n-by-1). The result is stored in <code>dest</code> (n-by-1),
	 * that is
	 * 
	 * <pre>
	 * dest = A * v
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param A
	 *            a quadratic <tt>n</tt>-by-<tt>n</tt> matrix
	 * @param v
	 *            a column vector of length <tt>n</tt>
	 * @param dest
	 *            this column vector of length <tt>n</tt> hols the product of
	 *            <tt>A</tt> and <tt>v</tt> on return
	 * @param n
	 *            size of the arrays to be multiplied
	 */
	public static void multiplyQuad(final double[][] A,
									final double[] v,
									final double[] dest,
									int n) {
		assert hasShape(A,n,n);
		assert v != null;
		assert dest != null;
		assert v.length >= n;
		assert dest.length >= n;
		
		for (int i = 0; i < n; i++) {
			dest[i] = A[i][0] * v[0];
			for (int j = 1; j < n; j++) {
				dest[i] += A[i][j] * v[j];
			}
		}
	}

	/**
	 * Multiplies the matrix <tt>A</tt> (l1-by-l2) with column vector
	 * <tt>v</tt> (l2-by-1). The result is stored in <code>dest</code>
	 * (l1-by-1), that is
	 * 
	 * <pre>
	 * dest = A * v
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param A
	 *            a <tt>l1</tt>-by-<tt>l2</tt> matrix
	 * @param v
	 *            a column vector of length <tt>l2</tt>
	 * @param dest
	 *            a column vector of length <tt>l1</tt> that holds the product
	 *            of <tt>A</tt> and <tt>v</tt> on return
	 * @param l1
	 *            number of rows of matrix A
	 * @param l2
	 *            number of columns of matrix A and length of column vector v
	 */
	public static void multiply(final double[][] A,
								final double[] v,
								final double[] dest,
								int l1,	int l2) {
		assert hasShape(A,l1,l2);
		assert v != null;
		assert dest != null;
		assert v.length >= l2;
		assert dest.length >= l1;
		
		for (int i = 0; i < l1; i++) {
			dest[i] = A[i][0] * v[0];
			for (int j = 1; j < l2; j++) {
				dest[i] += A[i][j] * v[j];
			}
		}
	}

	/**
	 * Multiplies the row vector <tt>v</tt> (1-by-n) with the quadratic matrix
	 * <tt>A</tt> (n-by-n). The result is stored in <code>dest</code> (1-by-n),
	 * that is
	 * 
	 * <pre>
	 * dest = v * A
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param vTransposed
	 *            a row vector of length <tt>n</tt>
	 * @param A
	 *            a quadratic <tt>n</tt>-by-<tt>n</tt> matrix
	 * @param destTransposed
	 *            this row vector of length <tt>n</tt> hols the product of
	 *            <tt>v</tt> and <tt>A</tt> on return
	 * @param n
	 *            size of the arrays to be multiplied
	 */
	public static void multiplyQuad(final double[] vTransposed,
									final double[][] A,
									final double[] destTransposed,
									int n) {
		assert vTransposed != null;
		assert destTransposed != null;
		assert vTransposed.length >= n;
		assert destTransposed.length >= n;
		assert hasShape(A,n,n);
		
		for (int j = 0; j < n; j++) {
			destTransposed[j] = vTransposed[0] * A[0][j];
			for (int i = 1; i < n; i++) {
				destTransposed[j] += vTransposed[i] * A[i][j];
			}
		}
	}

	/**
	 * Multiplies the row vector <tt>v</tt> (1-by-l1) with the quadratic matrix
	 * <tt>A</tt> (l1-by-l2). The result is stored in <code>dest</code> (1-by-l2),
	 * that is
	 * 
	 * <pre>
	 * dest = v * A
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param vTransposed
	 *            a row vector of length <tt>l1</tt>
	 * @param A
	 *            a <tt>l1</tt>-by-<tt>l2</tt> matrix
	 * @param destTransposed
	 *            this row vector of length <tt>l2</tt> hols the product of
	 *            <tt>v</tt> and <tt>A</tt> on return
	 * @param l1
	 *            number of rows of matrix A and length of row vector v
	 * @param l2
	 *            number of columns of matrix A and length of column vector dest
	 */
	public static void multiply(final double[] vTransposed,
								final double[][] A,
								final double[] destTransposed,
								int l1, int l2) {
		assert vTransposed != null;
		assert destTransposed != null;
		assert vTransposed.length >= l1;
		assert destTransposed.length >= l2;
		assert hasShape(A,l1,l2);
		
		for (int j = 0; j < l2; j++) {
			destTransposed[j] = vTransposed[0] * A[0][j];
			for (int i = 1; i < l1; i++) {
				destTransposed[j] += vTransposed[i] * A[i][j];
			}
		}
	}

	/**
	 * Multiplies the column vector <tt>v1</tt> (n-by-1) with the row vector
	 * <tt>v2</tt> (1-by-n). The result is stored in the quadratic matrix
	 * <code>dest</code> (n-by-n), that is
	 * 
	 * <pre>
	 * dest = v1 * v2
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param columnVector
	 *            a column vector of length <tt>n</tt>
	 * @param rowVector
	 *            a row vector of length <tt>n</tt>
	 * @param dest
	 *            this quadratic <tt>n</tt>-by-<tt>n</tt> matrix holds the
	 *            product of <code>columnVector</code> and
	 *            <code>rowVector</code> on return
	 * @param n
	 *            size of the arrays to be multiplied
	 */
	public static void multiply(final double[] columnVector,
								final double[] rowVector,
								final double[][] dest, int n) {
		assert columnVector != null;
		assert rowVector != null;
		assert dest != null;
		assert columnVector.length >= n;
		assert rowVector.length >= n;
		assert hasShape(dest,n,n);
		
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				dest[i][j] = columnVector[i] * rowVector[j];
			}
		}
	}
	
	/**
	 * In-place version of the vector product.
	 * 
	 * <pre>
	 * dest = v1 * v2
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param columnVector
	 *            a column vector of length <tt>n</tt>
	 * @param rowVector
	 *            a row vector of length <tt>n</tt>
	 * @param dest
	 *            this quadratic <tt>n</tt>-by-<tt>n</tt> matrix holds the
	 *            product of <code>columnVector</code> and
	 *            <code>rowVector</code> on return
	 * @param n
	 *            size of the arrays to be multiplied
	 */
	public static double[][] multiply(final double[] columnVector,
								final double[] rowVector,
								int n) {
		assert n > 0;
		
		final double[][] dest = new double[n][n];
		multiply(columnVector,rowVector,dest,n);
		return dest;
	}

	/**
	 * Multiplies two affine matrices <tt>A</tt> and <tt>B</tt> (n by n+1,
	 * linear part plus translation). The result is stored in <code>dest</code>,
	 * that is
	 * 
	 * <pre>
	 * dest = A * B
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param A
	 *            an affine <tt>n</tt>-by-<tt>n+1</tt> matrix
	 * @param B
	 *            an affine <tt>n</tt>-by-<tt>n+1</tt> matrix
	 * @param dest
	 *            this affine <tt>n</tt>-by-<tt>n+1</tt> matrix holds the
	 *            product of <tt>A</tt> and <tt>B</tt> on return
	 * @param n
	 *            dimension of matrices (without translational component)
	 */
	public static void multiplyAffine(final double[][] A,
									  final double[][] B,
									  final double[][] dest,
									  int n) {
		assert A != null;
		assert B != null;
		assert dest != null;
		assert hasShape(A,n,n+1);
		assert hasShape(B,n,n+1);
		assert hasShape(dest,n,n+1);
		
		for (int i = 0; i < n; i++) {
			// regular linear part
			for (int j = 0; j < n; j++) {
				dest[i][j] = A[i][0] * B[0][j];
				for (int k = 1; k < n; k++) {
					dest[i][j] += A[i][k] * B[k][j];
				}
			}
			// translational part
			dest[i][n] = A[i][n]; // * 1
			for (int k = 0; k < n; k++) {
				dest[i][n] += A[i][k] * B[k][n];
			}
		}
	}

	/**
	 * Multiplies the affine matrix <tt>A</tt> (n by n+1, linear part plus
	 * translation) with the column vector <tt>v</tt> (n-by-1) and adds the
	 * translational component of <tt>A</tt> to <tt>v</tt>. The result is stored
	 * in <code>dest</code>, that is
	 * 
	 * <pre>
	 * dest = (A * v) + translation
	 * </pre>
	 * 
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param A
	 *            an affine <tt>n</tt> by <tt>n+1</tt> matrix
	 * @param v
	 *            a column vector of length <tt>n</tt>
	 * @param dest
	 *            this column vector of length <tt>n</tt> holds the result of
	 *            the affine transformation
	 * @param n
	 *            size of the arrays to be multiplied (not including the
	 *            extended column/row)
	 */
	public static void multiplyAffine(final double[][] A,
									  final double[] v,
									  final double[] dest,
									  int n) {
		assert hasShape(A,n,n+1);
		assert v != null;
		assert dest != null;
		assert v.length >= n;
		assert dest.length >= n;
		
		for (int i = 0; i < n; i++) {
			dest[i] = A[i][n]; // translation
			for (int j = 0; j < n; j++) {
				dest[i] += A[i][j] * v[j]; // linear part
			}
		}
	}

	/**
	 * Computes the dotproduct {@code <v1,v2>} of the given vectors.
	 * 
	 * @param v1
	 *            first vector of length <tt>n</tt>
	 * @param v2
	 *            second vector of length <tt>n</tt>
	 * @param n
	 *            length of the vectors
	 * @return dot product of <tt>v1</tt> and <tt>v2</tt>
	 */
	public static double dotProduct(final double[] v1,
									final double[] v2,
									int n) {
		assert v1 != null;
		assert v2 != null;
		assert v1.length >= n;
		assert v2.length >= n;
		
		double dp = 0;
		for (int i = 0; i < n; i++) {
			dp += v1[i] * v2[i];
		}
		return dp;
	}

	/**
	 * Computes the element-wise product {@code v1*v2} of the given vectors.
	 * 
	 * @param v1
	 *            first vector of length <tt>n</tt>
	 * @param v2
	 *            second vector of length <tt>n</tt>
	 * @param result
	 *            vector of length <tt>n</tt> to store the element-wise product
	 * @param n
	 *            length of the vectors
	 */
	public static void elemProduct(final double[] v1,
								   final double[] v2,
								   final double[] result,
								   int n) {
		assert v1 != null;
		assert v2 != null;
		assert v1.length >= n;
		assert v2.length >= n;
		
		for(int i = 0; i < n; i++) {
			result[i] = v1[i] * v2[i];
		}
	}

	/**
	 * Computes the element-wise product {@code s*v} of scalar s with vector v.
	 * 
	 * @param s
	 *            scalar
	 * @param v
	 *            vector of length <tt>n</tt>
	 * @param result
	 *            vector of length <tt>n</tt> to store the element-wise product
	 * @param n
	 *            length of vector v
	 */
	public static void elemProduct(double s, final double[] v,
								   final double[] result, int n) {
		assert v != null;
		assert result != null;
		assert v.length >= n;
		assert result.length >= n;
		
		for(int i = 0; i < n; i++) {
			result[i] = s * v[i];
		}
	}

	/**
	 * Computes the element-wise product {@code s*A} of scalar s with matrix A.
	 * 
	 * @param s
	 *            scalar
	 * @param A
	 *            <tt>l1</tt> by <tt>l2</tt> matrix
	 * @param result
	 *            <tt>l1</tt> by <tt>l2</tt> matrix to store the element-wise product
	 * @param l1
	 *            number of rows of matrix A and result
	 * @param l2
	 *            number of columns of matrix A and result
	 */
	public static void elemProduct(double s, final double[][] A,
								   final double[][] result, int l1, int l2) {
		assert hasShape(A, l1, l2);
		assert hasShape(result, l1, l2);
		
		for(int i = 0; i < l1; i++) {
			for(int j = 0; j < l2; j++) {
				result[i][j] = s * A[i][j];
			}
		}
	}

	/**
	 * Computes the element-wise division {@code v1/v2} of the given vectors.
	 * 
	 * @param v1
	 *            first vector of length <tt>n</tt>
	 * @param v2
	 *            second vector of length <tt>n</tt>
	 * @param result
	 *            vector of length <tt>n</tt> to store the element-wise division
	 * @param n
	 *            length of the vectors
	 */
	public static void elemDiv(final double[] v1, final double[] v2,
							   final double[] result, int n) {
		assert v1 != null;
		assert v2 != null;
		assert result != null;
		assert v1.length >= n;
		assert v2.length >= n;
		assert result.length >= n;
		
		for(int i = 0; i < n; i++) {
			result[i] = v1[i] / v2[i];
		}
	}

	public static void elemSum(final double[] v1, final double[] v2,
							   final double[] result, int n) {
		assert v1 != null;
		assert v2 != null;
		assert result != null;
		assert v1.length >= n;
		assert v2.length >= n;
		assert result.length >= n;
		
		for(int i = 0; i < n; i++) {
			result[i] = v1[i] + v2[i];
		}
	}

	public static void elemSum(double[][] A, double[][] B,
			double[][] C, int l1, int l2) {
		assert hasShape(A, l1, l2);
		assert hasShape(B, l1, l2);
		assert hasShape(C, l1, l2);
		
		for(int i = 0; i < l1; i++) {
			for(int j = 0; j < l2; j++) {
				C[i][j] = A[i][j] + B[i][j];
			}
		}
	}

	public static void elemSubstract(final double[] v1,
									 final double[] v2,
									 final double[] result,
									 int n) {
		assert v1 != null;
		assert v2 != null;
		assert result != null;
		assert v1.length >= n;
		assert v2.length >= n;
		assert result.length >= n;
		
		for(int i = 0; i < n; i++) {
			result[i] = v1[i] - v2[i];
		}
	}

	public static void elemSubstract(final double[][] A,
									 final double[][] B,
									 final double[][] C,
									 int l1, int l2) {
		assert hasShape(A, l1, l2);
		assert hasShape(B, l1, l2);
		assert hasShape(C, l1, l2);
		
		for(int i = 0; i < l1; i++) {
			for(int j = 0; j < l2; j++) {
				C[i][j] = A[i][j] - B[i][j];
			}
		}
	}

	/**
	 * Copy the segment of src with indexes [pos, pos+size[ into source
	 * @param src     source vector
	 * @param pos     start position of the segment to copy
	 * @param length  length of the segment to copy
	 * @param dest    destination vector
	 */
	public static void segment(final double[] src, int pos,
							   int length, final double[] dest) {
		assert src != null;
		assert dest != null;
		assert src.length - pos >= length;
		assert dest.length >= length;
		
		System.arraycopy(src, pos, dest, 0, length);
	}

	/**
	 * Computes the quadratic form <tt>d²=x'Ax</tt>, which corresponds to the squared
	 * distance of a given point <tt>x</tt> from the center of an ellipsoid
	 * represented by the positive, definite matrix <tt>A</tt>.
	 * 
	 * @param x
	 *            a column vector of length <tt>n</tt>
	 * @param A
	 *            a n-by-n matrix
	 * @param n
	 *            size of the arrays to be multiplied
	 * @return the quadratic form
	 */
	public static double quadraticForm(final double[] x,
									   final double[][] A,
									   int n) {
		assert x != null;
		assert x.length >= n;
		assert hasShape(A, n, n);
		
		double q = 0, dp;
		for (int i = 0; i < n; i++) {
			dp = 0; // dp = <a[i],x>
			for (int j = 0; j < n; j++) {
				dp += A[i][j] * x[j];
			}
			q += x[i] * dp;
		}
		return q;
	}

	/**
	 * Computes the quadratic form <tt>d²=x'Ax</tt>, which corresponds to the squared
	 * distance of a given point <tt>x</tt> from the center of an ellipsoid
	 * represented by the diagonal matrix <tt>A</tt>.
	 * 
	 * @param x
	 *            a column vector of length <tt>n</tt>
	 * @param a
	 *            the <tt>n</tt> diagonal elements of matrix A
	 * @param n
	 *            size of the arrays to be multiplied
	 * @return the quadratic form
	 */
	public static double quadraticForm(final double[] x,
									   final double[] a,
									   int n) {
		assert x != null;
		assert a != null;
		assert x.length >= n;
		assert a.length >= n;
		
		double q = 0;
		for (int i = 0; i < n; i++) {
			q += x[i] * a[i] * x[i];
		}
		return q;
	}

	/**
	 * Computes the n by n rotation matrix corresponding to the given angles.
	 * 
	 * @param angles
	 *            the <tt>n*(n-1)/2</tt> rotation angles for the principal
	 *            rotation planes
	 * @param rotation
	 *            quadratic, <tt>n</tt> by <tt>n</tt> matrix that holds the
	 *            rotation on return
	 * @param n
	 *            dimension of the rotation matrix
	 */
	public static void computeRotationMatrix(final double[] angles,
											 final double[][] rotation,
											 int n) {
		assert hasShape(rotation,n,n);
		assert angles != null;
		assert angles.length == n * (n - 1) / 2 : "bad number of angles";
	
		// initialize as identity
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				rotation[i][j] = i == j ? 1 : 0;
			}
		}
		// multiply with individual rotations.
		int index = 0; // index for angles array
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				// plane of rotation is i/j
				ArrUtils.applySingleRotation(rotation, n, angles[index++], i, j);
			}
		}
	}

	/**
	 * Applies a single rotation to the given matrix about the given angle. The
	 * plane of rotation is the plane spanned by axis <tt>i</tt> and axis
	 * <tt>j</tt>; orthogonal dimensions are not affected by this rotation. The
	 * rotation angle is <tt>theta</tt>. Note that switching the indices results
	 * in the inverse rotation, which also results from using the inverse angle.
	 * 
	 * @param A
	 *            a quadratic <tt>n</tt> by <tt>n</tt> matrix to apply the
	 *            rotation to
	 * @param n
	 *            the dimension of the matrix <tt>a</tt>
	 * @param theta
	 *            the angle to rotate (given in radians)
	 * @param i
	 *            specifies the first axis of the rotation plane
	 * @param j
	 *            specifies the second axis of the rotation plane
	 */
	public static void applySingleRotation(final double[][] A, int n,
										   double theta, int i, int j) {
		assert hasShape(A,n,n);
		assert i != j;
		assert i >= 0;
		assert j >= 0;
		assert i < n;
		assert j < n;
	
		// zero angle? Note: checking for theta%2pi==0 doesn't work
		if (theta == 0) {
			return;
		}
		double c = Math.cos(theta);
		double s = Math.sin(theta);
		// only the columns i and j are affected
		for (int row = 0; row < n; row++) {
			// column i
			double ari = A[row][i] * c + A[row][j] * s;
			// column j
			double arj = A[row][i] * -s + A[row][j] * c;
			A[row][i] = ari;
			A[row][j] = arj;
		}
	}

	/**
	 * Multiplies the matrix <tt>V</tt> (l1-by-l2) with diagonal matrix <tt>D</tt> (l2-by-l2) and
	 * then with vector <tt>w</tt> (l2). The result is stored in the vector <tt>A</tt> (l1),
	 * that is
	 * 
	 * <pre>
	 * A = V * D * w
	 * </pre>
	 * 
	 * If all diagonal entries of <tt>D</tt> are greater than zero, the
	 * resulting matrix is also positive definite.
	 * <p>
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param V
	 *            a quadratic <tt>l1</tt>-by-<tt>l2</tt> matrix
	 * @param d
	 *            the <tt>l2</tt> diagonal entries of a (diagonal) matrix
	 * @param w
	 *            a vector of length <tt>l2</tt>
	 * @param a
	 *            on return, contains <tt>V*D*w</tt> which is a vector of length <tt>l1</tt>
	 * @param l1
	 *            number of rows of <tt>V</tt>
	 * @param l2
	 *            number of columns of <tt>V</tt>
	 */
	public static void matrixVectorProductVDw(final double[][] V,
											  final double[] d,
											  final double[] w,
											  final double[] a,
											  int l1, int l2) {
		assert hasShape(V, l1, l2);
		assert d != null;
		assert w != null;
		assert a != null;
		assert d.length >= l2;
		assert w.length >= l2;
		assert a.length >= l1;
		
		for (int i = 0; i < l1; i++) {
			// V D w
			a[i] = V[i][0] * d[0] * w[0];
			for (int j = 1; j < l2; j++) {
				a[i] += V[i][j] * d[j] * w[j];
			}
		}
	}

	/**
	 * Multiplies the vector <tt>v</tt> (n) with diagonal matrix <tt>D</tt> (n-by-n) and then with
	 * <tt>v'</tt> (transposed). The result is stored in the symmetric matrix <tt>A</tt> (n-by-n),
	 * that is
	 * 
	 * <pre>
	 * A = v * D * v'
	 * </pre>
	 * 
	 * If all diagonal entries of <tt>D</tt> are greater than zero, the
	 * resulting matrix is also positive definite.
	 * <p>
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param V
	 *            a quadratic <tt>n</tt>-by-<tt>n</tt> matrix
	 * @param d
	 *            the <tt>n</tt> diagonal entries of a (diagonal) matrix
	 * @param A
	 *            on return, contains <tt>V*D*V'</tt> which is symmetric and, if all
	 *            <tt>d[i] > 0</tt>, positive definite
	 * @param n
	 *            size of the arrays to be multiplied
	 */
	public static void matrixProductVDVT(final double[][] V,
										 final double[] d,
										 final double[][] A,
										 int n) {
		assert hasShape(V, n, n);
		assert d != null;
		assert d.length >= n;
		assert hasShape(A, n, n);
		
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				// V D V^T
				A[i][j] = V[i][0] * d[0] * V[j][0];
				for (int k = 1; k < n; k++) {
					A[i][j] += V[i][k] * d[k] * V[j][k];
				}
				// A is symmetric
				A[j][i] = A[i][j];
			}
		}
	}

	/**
	 * Multiplies the matrix <tt>V</tt> (l1-by-l2) with diagonal matrix
	 * <tt>D</tt> (l2-by-l2) and then with <tt>V'</tt> (transposed). The result is
	 * stored in the symmetric matrix <tt>A</tt> (l1-by-l1), that is
	 * 
	 * <pre>
	 * A = V * D * V'
	 * </pre>
	 * 
	 * If all diagonal entries of <tt>D</tt> are greater than zero, the
	 * resulting matrix is also positive definite.
	 * <p>
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param v
	 *            a <tt>l1</tt>-by-<tt>l2</tt> matrix
	 * @param d
	 *            the <tt>l2</tt> diagonal entries of a (diagonal) matrix
	 * @param a
	 *            on return, contains <tt>V*D*V'</tt> which is symmetric and, if all
	 *            <tt>d[i] > 0</tt>, positive definite
	 * @param l1
	 *            number of rows of <tt>V</tt>
	 * @param l2
	 * 			  number of columns of <tt>V</tt> and number of rows and columns of <tt>D</tt>
	 */
	public static void matrixProductVDVT(final double[][] V,
										 final double[] d,
										 final double[][] A,
										 int l1, int l2) {
		assert hasShape(V, l1, l2);
		assert d != null;
		assert d.length >= l2;
		assert hasShape(A, l1, l1);
		
		for (int i = 0; i < l1; i++) {
			for (int j = i; j < l1; j++) {
				// V D V^T
				A[i][j] = V[i][0] * d[0] * V[j][0];
				for (int k = 1; k < l2; k++) {
					A[i][j] += V[i][k] * d[k] * V[j][k];
				}
				// A is symmetric
				A[j][i] = A[i][j];
			}
		}
	}

	/**
	 * Multiplies the matrix <tt>V</tt> (l1-by-l2) with diagonal matrix
	 * <tt>D</tt> (l2-by-l2) and then with the transposed of matrix <tt>W</tt> (l3-by-l2).
	 * The result is stored in the symmetric matrix <tt>A</tt> (l1-by-l3), that is
	 * 
	 * <pre>
	 * A = V * D * W'
	 * </pre>
	 * 
	 * <p>
	 * Note that this method does no safety checks (null or length).
	 * 
	 * @param V
	 *            a <tt>l1</tt>-by-<tt>l2</tt> matrix
	 * @param d
	 *            the <tt>l2</tt> diagonal entries of a (diagonal) matrix
	 * @param W
	 *            a <tt>l3</tt>-by-<tt>l2</tt> matrix
	 * @param A
	 *            on return, contains the <tt>l1</tt>-by-<tt>l3</tt> matrix <tt>V*D*W'</tt>
	 * @param l1
	 *            number of rows of <tt>V</tt>
	 * @param l2
	 * 			  number of columns of <tt>V</tt> and number of rows and columns of <tt>D</tt>
	 */
	public static void matrixProductVDVT(final double[][] V,
										 final double[] d,
										 final double[][] W,
										 final double[][] A,
										 int l1, int l2, int l3) {
		assert hasShape(V, l1, l2);
		assert d != null;
		assert d.length >= l2;
		assert hasShape(W, l3, l2);
		assert hasShape(A, l1, l3);
		
		for (int i = 0; i < l1; i++) {
			for (int j = 0; j < l3; j++) {
				// V D W^T
				A[i][j] = V[i][0] * d[0] * V[j][0];
				for (int k = 1; k < l2; k++) {
					A[i][j] += V[i][k] * d[k] * V[j][k];
				}
			}
		}
	}

	/**
	 * Transposes the given quadratic matrix in place.
	 * 
	 * @param matrix
	 *            the quadratic <tt>n</tt>-by-<tt>n</tt> matrix to be transposed
	 * @param n
	 *            size of the matrix
	 */
	public static void transposeQuadInPlace(final double[][] matrix, int n) {
		assert hasShape(matrix, n, n);
		
		double tmp;
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				tmp = matrix[i][j];
				matrix[i][j] = matrix[j][i];
				matrix[j][i] = tmp;
			}
		}
	}

	/**
	 * Transposes the source matrix in the destination matrix.
	 * 
	 * @param src
	 *            the source <tt>l1</tt>-by-<tt>l2</tt> matrix to be transposed
	 * @param dest
	 *            the destination <tt>l2</tt>-by-<tt>l1</tt> matrix,
	 *            transpose of src
	 * @param l1
	 *            number of rows of source matrix
	 * @param l2
	 *            number of columns of source matrix
	 */
	public static void transpose(final double[][] src,
								 final double[][] dest,
								 int l1, int l2) {
		assert src != null;
		assert dest != null;
		assert hasShape(src, l1, l2);
		assert hasShape(dest, l2, l1);
		
		for (int i = 0; i < l1; i++) {
			for (int j = 0; j < l2; j++) {
				dest[j][i] = src[i][j];
			}
		}
	}

	/**
	 * Transposes the given quadratic matrix.
	 * 
	 * @param src
	 *            the quadratic <tt>n</tt>-by-<tt>n</tt> matrix to be transposed
	 * @param n
	 *            size of the matrix
	 * @return a new <tt>n</tt>-by-<tt>n</tt> matrix,
	 *         transpose of src
	 */
	public static double[][] transposeQuad(
			final double[][] src, int n) {
		assert hasShape(src, n, n);
		
		final double[][] dest = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				dest[i][j] = src[j][i];
				dest[j][i] = src[i][j];
			}
		}
		return dest;
	}

	/**
	 * Transposes the source matrix in the destination matrix.
	 * 
	 * @param src
	 *            the source <tt>l1</tt>-by-<tt>l2</tt> matrix to be transposed
	 * @param l1
	 *            number of rows of source matrix
	 * @param l2
	 *            number of columns of source matrix
	 * @return a new <tt>l2</tt>-by-<tt>l1</tt> matrix,
	 *         transpose of src
	 */
	public static final double[][] transpose(
			final double[][] src,
			int l1, int l2) {
		assert src != null;
		assert hasShape(src, l1, l2);
		
		final double[][] dest = new double[l2][l1];
		for (int i = 0; i < l1; i++) {
			for (int j = 0; j < l2; j++) {
				dest[j][i] = src[i][j];
			}
		}
		return dest;
	}
	
	/**
	 * Returns a new matrix exactly like a given matrix.
	 * 
	 * @param src
	 *            the <tt>l1</tt>-by-<tt>l2</tt> matrix to be copied
	 * @param l1
	 *            number of rows of source matrix
	 * @param l2
	 *            number of columns of source matrix
	 * @return dest
	 *            the <tt>l1</tt>-by-<tt>l2</tt> matrix cloned from the source
	 */
	public static double[][] cloneMatrix(final double[][] src,
												  int l1, int l2) {
		assert hasShape(src, l1, l2);
		
		final double[][] mat = empty(l1,l2);
		copyMatrix(src, mat, l1, l2);
		return mat;
	}

	/**
	 * Copies a quadratic matrix from one array to another without any safety
	 * checks.
	 * 
	 * @param src
	 *            the quadratic <tt>n</tt>-by-<tt>n</tt> matrix to be copied
	 * @param dest
	 *            the quadratic <tt>n</tt>-by-<tt>n</tt> matrix to hold the new
	 *            copy
	 * @param n
	 *            number of rows and columns of source matrix
	 */
	public static void copyMatrix(final double[][] src,
								  final double[][] dest,
								  int n) {
		assert hasShape(src, n, n);
		assert hasShape(dest, n, n);
		assert src != dest;
		
		for (int i = 0; i < n; i++) {
			System.arraycopy(src[i], 0, dest[i], 0, n);
		}
	}

	/**
	 * Copies a matrix from one array to another without any safety checks.
	 * 
	 * @param src
	 *            the <tt>l1</tt>-by-<tt>l2</tt> matrix to be copied
	 * @param dest
	 *            the <tt>l1</tt>-by-<tt>l2</tt> matrix to hold the new copy
	 * @param l1
	 *            number of rows of source matrix
	 * @param l2
	 *            number of columns of source matrix
	 */
	public static void copyMatrix(final double[][] src,
								  final double[][] dest,
								  int l1, int l2) {
		assert hasShape(src, l1, l2);
		assert hasShape(dest, l1, l2);
		
		for (int i = 0; i < l1; i++) {
			System.arraycopy(src[i], 0, dest[i], 0, l2);
		}
	}

	/**
	 * Copies a block from one matrix to another
	 * @param src     source matrix
	 * @param i_src   first row of the source matrix block to copy
	 * @param j_src   first column of the source matrix block to copy
	 * @param dest    destination matrix
	 * @param i_dest  first row of the destination matrix block to be copied to
	 * @param j_dest  first column of the destination matrix block to be copied to
	 * @param rows    number of rows to copy
	 * @param cols    number of columns to copy
	 */
	public static void copyMatrixBlock(final double[][] src, int i_src, int j_src,
									   final double[][] dest, int i_dest, int j_dest,
									   int rows, int cols) {
		assert src != null;
		assert dest != null;
		assert src != dest;
		// TODO do all the other safety checks
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				dest[i_dest+i][j_dest+j] = src[i_src+i][j_src+j];
			}
		}
	}

	/**
	 * Cholesky decomposition of the symmetric, positive-definite matrix
	 * <tt>A</tt>. The returned lower-diagonal matrix <tt>G</tt> represents the
	 * matrix factor, that is
	 * 
	 * <pre>
	 * A = G * G'
	 * </pre>
	 * 
	 * No safety checks are done, unless assertions are enabled (run java with
	 * the <tt>-enableassertions</tt> or <tt>-ea</tt> switch).
	 * 
	 * @param A
	 *            a symmetric, positive definite <tt>n</tt>-by-<tt>n</tt> matrix
	 *            to be decomposed
	 * @param dest
	 *            [on return] the lower-diagonal <tt>n</tt>-by-<tt>n</tt> matrix
	 *            factor <tt>G</tt>
	 * @param n
	 *            dimension of the matrices
	 */
	public static void choleskyDecomposition(final double[][] A,
											 final double[][] dest,
											 int n) {
		assert isSymmetric(A);
		assert hasShape(dest,n,n); 
		assert A.length == n;
	
		// prepare work space "dest"
		for (int i = 0; i < n; i++) {
			// copy lower-triangular part
			for (int j = 0; j <= i; j++) {
				dest[i][j] = A[i][j];
			}
			// upper triangle is 0
			for (int j = i + 1; j < n; j++) {
				dest[i][j] = 0.0;
			}
		}
		choleskyDecompositionInPlace(dest, n);
	}

	/**
	 * Cholesky decomposition of the symmetric, positive-definite matrix
	 * <tt>A</tt> in place. The upper diagonal entries of the matrix are not
	 * used (the matrix is symmetric) and the method will not touch them. On
	 * return the matrix <tt>G</tt> contains the lower-diagonal matrix factor,
	 * that is
	 * 
	 * <pre>
	 * A = G * G'
	 * </pre>
	 * 
	 * @param A
	 *            [on call] this matrix contains the diagonal and the lower
	 *            diagonal entries of the symmetric, positive definite matrix
	 *            <tt>A</tt><br/>
	 *            [on return] contains the lower diagonal matrix factor
	 *            <tt>G</tt>.
	 * @param n
	 *            the dimension of the matrix
	 */
	public static void choleskyDecompositionInPlace(final double[][] A,
													int n) {
		assert A != null;
		assert A.length == n;
		assert isQuadratic(A);
	
		double sum;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < i; j++) {
				sum = A[i][j];
				for (int k = 0; k < j; k++) {
					sum -= A[i][k] * A[j][k];
				}
				A[i][j] = sum / A[j][j];
			}
			sum = A[i][i];
			for (int k = 0; k < i; k++) {
				sum -= A[i][k] * A[i][k];
			}
			if (sum <= 0) {
				return;
			}
			A[i][i] = Math.sqrt(sum);
		}
	}

	/**
	 * Computes the eigendecomposition of a symmetric matrix <tt>A</tt>. The
	 * decomposition is given by
	 * 
	 * <pre>
	 * A = V * D * V'
	 * </pre>
	 * 
	 * where <tt>D</tt> is a diagonal matrix that contains the eigenvalues and
	 * the columns of <tt>V</tt> are the orthogonal eigenvectors. Use
	 * {@link ArrUtils#constructDfromEigenDecomposition(double[], double[], double[][], int)}
	 * to get the complete matrix <tt>D</tt>.
	 * <p>
	 * No safety checks are done, unless assertions are enabled (run java with
	 * the <tt>-enableassertions</tt> or <tt>-ea</tt> switch).
	 * <p>
	 * This is derived from the Algol procedures tred2 by Bowdler, Martin,
	 * Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra,
	 * and the corresponding Fortran subroutine in EISPACK.
	 * 
	 * @param A
	 *            the symmetric (<tt>n</tt> by <tt>n</tt>) matrix to be
	 *            decomposed
	 * @param V
	 *            [on return] orthogonal eigenvectors of matrix <tt>A</tt>
	 * @param re
	 *            [on return] real eigenvalues
	 * @param ie
	 *            [on return] imaginary eigenvalues
	 * @param n
	 *            the dimension of the matrix <tt>A</tt>
	 * @see ArrUtils#constructDfromEigenDecomposition(double[], double[], double[][],
	 *      int)
	 */
	public static void eigenDecomposition(final double[][] A,
										  final double[][] V,
										  final double[] re,
										  final double[] ie,
										  int n) {
		assert isSymmetric(A);
		assert re != null;
		assert ie != null;
		assert re != ie;
		assert A.length == n;
		assert V.length == n;
		assert re.length == n;
		assert ie.length == n;
	
		// copy A to V and do the decomposition in place
		for (int i = 0; i < n; i++) {
			System.arraycopy(A[i], 0, V[i], 0, n);
		}
		tred2(V, re, ie, n);
		tql2(V, re, ie, n);
	}

	/**
	 * Computes the eigendecomposition of a symmetric matrix <tt>A</tt> in
	 * place. The decomposition is given by
	 * 
	 * <pre>
	 * A = V*D*V'
	 * </pre>
	 * 
	 * where <tt>D</tt> is a diagonal matrix that contains the eigenvalues and
	 * the columns of <tt>V</tt> are the orthogonal eigenvectors. Use
	 * {@link ArrUtils#constructDfromEigenDecomposition(double[], double[], double[][], int)}
	 * to get the complete matrix <tt>D</tt>.
	 * <p>
	 * No safety checks are done, unless assertions are enabled (run java with
	 * the <tt>-enableassertions</tt> or <tt>-ea</tt> switch).
	 * <p>
	 * This is derived from the Algol procedures tred2 by Bowdler, Martin,
	 * Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra,
	 * and the corresponding Fortran subroutine in EISPACK.
	 * 
	 * @param v
	 *            [on call] the symmetric, positive definite (<tt>n</tt> by
	 *            <tt>n</tt>) matrix <tt>A</tt><br/>
	 *            [on return] orthogonal eigenvectors of matrix <tt>A</tt>
	 * @param re
	 *            [on return] real eigenvalues
	 * @param ie
	 *            [on return] imaginary eigenvalues
	 * @param n
	 *            the dimension of the matrix <tt>A</tt>
	 * @see ArrUtils#constructDfromEigenDecomposition(double[], double[], double[][],
	 *      int)
	 */
	public static void eigenDecompositionInPlace(final double[][] V,
												 final double[] re,
												 final double[] ie,
												 int n) {
		assert re != null;
		assert ie != null;
		assert re.length == n;
		assert ie.length == n;
		assert isQuadratic(V);
		assert V.length == n;
	
		ArrUtils.tred2(V, re, ie, n);
		ArrUtils.tql2(V, re, ie, n);
	}

	/**
	 * Constructs the diagonal matrix <tt>D</tt> from real and imaginary
	 * eigenvalues of an eigendecomposition.
	 * <p>
	 * No safety checks are done, unless assertions are enabled (run java with
	 * the <tt>-enableassertions</tt> or <tt>-ea</tt> switch).
	 * 
	 * @param re
	 *            the real eigenvalues
	 * @param ie
	 *            the imaginary eigenvalues
	 * @param D
	 *            [on return] the diagonal matrix <tt>D</tt>
	 * @param n
	 *            the dimension of the arrays and the matrix
	 * @see eigenDecomposition
	 */
	public static void constructDfromEigenDecomposition(final double[] re,
														final double[] ie,
														final double[][] D,
														int n) {
		assert re != null;
		assert ie != null;
		assert re != ie;
		assert re.length >= n;
		assert ie.length >= n;
		assert hasShape(D,n,n);
	
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				D[i][j] = 0.0;
			}
			D[i][i] = re[i];
			if (ie[i] > 0) {
				D[i][i + 1] = ie[i];
			} else if (ie[i] < 0) {
				D[i][i - 1] = ie[i];
			}
		}
	}

	/**
	 * Constructs the diagonal matrix <tt>D</tt> from real eigenvalues of an
	 * eigendecomposition.
	 * <p>
	 * No safety checks are done, unless assertions are enabled (run java with
	 * the <tt>-enableassertions</tt> or <tt>-ea</tt> switch).
	 * 
	 * @param re
	 *            the real eigenvalues
	 * @param D
	 *            [on return] the diagonal matrix <tt>D</tt>
	 * @param n
	 *            the dimension of the arrays and the matrix
	 * @see eigenDecomposition
	 */
	public static void constructDfromEigenDecomposition(final double[] re,
														final double[][] D,
														int n) {
		assert re != null;
		assert re.length >= n;
		assert hasShape(D,n,n);
	
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				D[i][j] = 0.;
			}
			D[i][i] = re[i];
		}
	}
	
	/**
	 * Computes the square root of a symmetric matrix <tt>A</tt> in place.
	 * @param mat
	 * 				[on call] the symmetric matrix <tt>A</tt>
	 * 				[on return] the root square matrix <tt>R</tt> of <tt>A</tt>
	 * @param n
	 * 				the dimension of the matrices <tt>A</tt> and <tt>R</tt> 
	 * @throws Exception 
	 */
	public static void squareRootInPlace(final double[][] A,
										 int n)
			                             throws Exception {
		assert hasShape(A, n, n);
		
		if(isSymmetric(A)) {
			final double[] re = new double[n];
			final double[] ie = new double[n];
			final double[][] V = new double[n][n];
			final double[][] D = new double[n][n];
			// Compute V and D such that mat = V * D * V^T
			// with D a diagonal
			eigenDecomposition(A, V, re, ie, n);
			constructDfromEigenDecomposition(re, D, n);
			// Compute the square root S of D
			for (int i = 0; i < D.length; i++) {
				D[i][i] = Math.sqrt(D[i][i]);
			}
			// Compute V^-1
			final double[][] Vinv = pinv(new Matrix(V)).getArray();
			if(Vinv == null) {
				throw new Exception("Vinv is null");
			}
			// Compute R = V * S * V^-1
			final double[][] tmp = new double[n][n];
			multiplyQuad(V, D, tmp, n);
			multiplyQuad(tmp, Vinv, A, n);
		} else {
			throw new UnsupportedOperationException();
			// FIXME not working
			//EigenvalueDecomposition eigA = new Matrix(A).eig();
			//Matrix V = eigA.getV();
			//Matrix D = eigA.getD();
			//Matrix Vinv = pinv(V);
			//Utils.copyMatrix(V.times(D).times(Vinv).getArray(), A, n);
		}
	}

	/**
	 * Computes the inverse of the symmetric positive definite quadratic matrix
	 * <tt>A</tt>. IMPORTANT: This method is rather slow and could be heavily
	 * optimized. Currently its only used for debugging purposes.
	 * 
	 * @param a
	 *            <tt>n</tt> by <tt>n</tt> symmetric positive definite matrix
	 * @param dest
	 *            <tt>n</tt> by <tt>n</tt> matrix; holds the inverse on return
	 * @param n
	 *            size of the matrices
	 */
	public static void slowInverse(final double[][] src,
								   final double[][] dest,
								   int n) {
		assert isSymmetric(src);
		assert src.length == n;
		assert hasShape(dest, n, n);
		assert src != dest;
	
		double[][] identity = new double[n][n];
		for (int i = 0; i < n; i++) {
			identity[i][i] = 1.0;
		}
	
		double[][] inverse = new LUDecomposition(src, n).solve(identity);
	
		for (int i = 0; i < n; i++) {
			System.arraycopy(inverse[i], 0, dest[i], 0, n);
		}
	}

	/**
	 * Copy the inverse of the 2-by-2 matrix src intro 2-by-2 matrix dest 
	 * 
	 * @param src
	 *            <tt>2</tt> by <tt>2</tt> matrix
	 * @param dest
	 *            <tt>2</tt> by <tt>2</tt> matrix to store the inverse of src
	 * @throws Exception 
	 * 
	 */
	public static void invert2x2(final double[][] src,
								 final double[][] dest)
								 throws Exception {
		assert hasShape(src, 2, 2);
		assert hasShape(dest, 2, 2);
		assert src != dest;
		
		final double tmp = src[0][0]*src[1][1]-src[0][1]*src[1][0];
		if(tmp == 0.) {
			throw new Exception();
		}
		dest[0][0] = src[1][1] / tmp;
		dest[1][1] = src[0][0] / tmp;
		dest[0][1] = -src[0][1] / tmp;
		dest[1][0] = -src[1][0] / tmp;
	}

	/**
	 * Computes Moore–Penrose pseudoinverse using the SVD method.
	 * Modified version of the original implementation by Kim van der Linde.
	 * @throws Exception 
	 */
	public static Matrix pinv(final Matrix x)
			                           throws Exception {
		assert x != null;
		
		if (x.rank() < 1) {
			throw new Exception("Invalid rank");
		}
		SingularValueDecomposition svdX = new SingularValueDecomposition(x);
		final double[] singularValues = svdX.getSingularValues();
		final double tol = Math.max(x.getColumnDimension(), x.getRowDimension()) * singularValues[0] * Utils.MACHEPS;
		final double[] singularValueReciprocals = new double[singularValues.length];
		for(int i = 0; i < singularValues.length; i++)
			singularValueReciprocals[i] = Math.abs(singularValues[i]) < tol ? 0 : (1.0 / singularValues[i]);
		final double[][] u = svdX.getU().getArray();
		final double[][] v = svdX.getV().getArray();
		final int min = Math.min(x.getColumnDimension(), u[0].length);
		final double[][] inverse = new double[x.getColumnDimension()][x.getRowDimension()];
		for (int i = 0; i < x.getColumnDimension(); i++)
			for (int j = 0; j < u.length; j++)
				for (int k = 0; k < min; k++)
					inverse[i][j] += v[i][k] * singularValueReciprocals[k] * u[j][k];
		return new Matrix(inverse);
	}

	/**
	 * Applies the Sherman-Morrison formula to update the inverse matrix Ainv
	 * with vectors u and v :
	 *     Ainv -= (Ainv * z * v^T * Ainv) / (1 + v^T * Ainv * u)
	 * which is equivalent to
	 *     A += u * v^T
	 * Fails if (1 + v^T * Ainv * u) is 0.
	 */
	public static void shermanMorrisonFormula(final double[][] Ainv,
											  final double[] u,
											  final double[] v,
											  int n)
											  throws Exception {
		assert hasShape(Ainv,n,n);
		assert u != null;
		assert v != null;
		assert u.length == n;
		assert v.length == n;
		
		final double[] tmp1 = new double[n];
		final double[][] tmp2 = new double[n][n];
		final double[][] tmp3 = new double[n][n];
		multiplyQuad(Ainv, u, tmp1, n);
	    multiply(tmp1, v, tmp2, n);
	    multiplyQuad(tmp2, Ainv, tmp3, n);
	    multiplyQuad(v, Ainv, tmp1, n);
	    final double tmpScal = dotProduct(tmp1, u, n);
	    if(tmpScal == 0.) {
	    	throw new Exception("Division by zero");
	    }
	    for(int i=0; i<n; i++) {
	    	for(int j=0; j<n; j++) {
	    		Ainv[i][j] -= tmp3[i][j] / (1.+tmpScal);
	    	}
	    }
	}

	public static boolean isQuadratic(final double[][] mat) {
		assert mat != null;
		final int n = mat.length;
		return hasShape(mat, n, n);
	}

	public static boolean isSymmetric(final double[][] mat) {
		assert mat != null;
		final int n = mat.length;
		assert hasShape(mat,n,n);
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				if (mat[i][j] != mat[j][i]) {
					return false;
				}
			}
		}
		return true;
	}

	/**
	 * Symmetric Householder reduction to tridiagonal form.
	 * <p>
	 * This is derived from the Algol procedures tred2 by Bowdler, Martin,
	 * Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra,
	 * and the corresponding Fortran subroutine in EISPACK.
	 * 
	 * @param V
	 *            [on return] eigen vectors
	 * @param re
	 *            [on return] real eigenvalues
	 * @param ie
	 *            [on return] imaginary eigenvalues
	 * @param n
	 */
	static void tred2(final double[][] V, final double[] re,
					  final double[] ie, int n) {
		assert hasShape(V,n,n);
		assert re != null;
		assert ie != null;
		assert re.length >= n;
		assert ie.length >= n;
		assert re != ie;
		
		for (int j = 0; j < n; j++) {
			re[j] = V[n - 1][j];
		}
	
		// Householder reduction to tridiagonal form.
		for (int i = n - 1; i > 0; i--) {
			// Scale to avoid under/overflow.
			double scale = 0.0;
			double h = 0.0;
			for (int k = 0; k < i; k++) {
				scale = scale + Math.abs(re[k]);
			}
			if (scale == 0.0) {
				ie[i] = re[i - 1];
				for (int j = 0; j < i; j++) {
					re[j] = V[i - 1][j];
					V[i][j] = 0.0;
					V[j][i] = 0.0;
				}
			} else {
				// Generate Householder vector.
				for (int k = 0; k < i; k++) {
					re[k] /= scale;
					h += re[k] * re[k];
				}
				double f = re[i - 1];
				double g = Math.sqrt(h);
				if (f > 0) {
					g = -g;
				}
				ie[i] = scale * g;
				h = h - f * g;
				re[i - 1] = f - g;
				for (int j = 0; j < i; j++) {
					ie[j] = 0.0;
				}
	
				// Apply similarity transformation to remaining columns.
				for (int j = 0; j < i; j++) {
					f = re[j];
					V[j][i] = f;
					g = ie[j] + V[j][j] * f;
					for (int k = j + 1; k <= i - 1; k++) {
						g += V[k][j] * re[k];
						ie[k] += V[k][j] * f;
					}
					ie[j] = g;
				}
				f = 0.0;
				for (int j = 0; j < i; j++) {
					ie[j] /= h;
					f += ie[j] * re[j];
				}
				double hh = f / (h + h);
				for (int j = 0; j < i; j++) {
					ie[j] -= hh * re[j];
				}
				for (int j = 0; j < i; j++) {
					f = re[j];
					g = ie[j];
					for (int k = j; k <= i - 1; k++) {
						V[k][j] -= (f * ie[k] + g * re[k]);
					}
					re[j] = V[i - 1][j];
					V[i][j] = 0.0;
				}
			}
			re[i] = h;
		}
		
		// Accumulate transformations.
		for (int i = 0; i < n - 1; i++) {
			V[n - 1][i] = V[i][i];
			V[i][i] = 1.0;
			final double h = re[i + 1];
			if (h != 0.0) {
				for (int k = 0; k <= i; k++) {
					re[k] = V[k][i + 1] / h;
				}
				for (int j = 0; j <= i; j++) {
					double g = 0.0;
					for (int k = 0; k <= i; k++) {
						g += V[k][i + 1] * V[k][j];
					}
					for (int k = 0; k <= i; k++) {
						V[k][j] -= g * re[k];
					}
				}
			}
			for (int k = 0; k <= i; k++) {
				V[k][i + 1] = 0.0;
			}
		}
		for (int j = 0; j < n; j++) {
			re[j] = V[n - 1][j];
			V[n - 1][j] = 0.0;
		}
		V[n - 1][n - 1] = 1.0;
		ie[0] = 0.0;
	}

	/**
	 * Symmetric tridiagonal QL algorithm.
	 * <p>
	 * This is derived from the Algol procedures tql2, by Bowdler, Martin,
	 * Reinsch, and Wilkinson, Handbook for Auto. Comp., Vol.ii-Linear Algebra,
	 * and the corresponding Fortran subroutine in EISPACK.
	 * 
	 * @param V
	 * @param re
	 * @param ie
	 * @param n
	 */
	static void tql2(final double[][] V, final double[] re,
					 final double[] ie, int n) {
		assert hasShape(V,n,n);
		assert re != null;
		assert ie != null;
		assert re.length >= n;
		assert ie.length >= n;
		assert re != ie;
		
		for (int i = 1; i < n; i++) {
			ie[i - 1] = ie[i];
		}
		ie[n - 1] = 0.0;
	
		double f = 0.0;
		double tst1 = 0.0;
		for (int l = 0; l < n; l++) {
	
			// Find small subdiagonal element
			tst1 = Math.max(tst1, Math.abs(re[l]) + Math.abs(ie[l]));
			int m = l;
			while (m < n) {
				if (Math.abs(ie[m]) <= EPSILON * tst1) {
					break;
				}
				m++;
			}
	
			// If m == l, d[l] is an eigenvalue,
			// otherwise, iterate.
			if (m > l) {
				do {
					// Compute implicit shift
					double g = re[l];
					double p = (re[l + 1] - g) / (2.0 * ie[l]);
					double r = Math.hypot(p, 1.0);
					if (p < 0) {
						r = -r;
					}
					re[l] = ie[l] / (p + r);
					re[l + 1] = ie[l] * (p + r);
					double dl1 = re[l + 1];
					double h = g - re[l];
					for (int i = l + 2; i < n; i++) {
						re[i] -= h;
					}
					f = f + h;
	
					// Implicit QL transformation.
					if (m >= n) {
						throw new IllegalArgumentException();
					}
					p = re[m];
					double c = 1.0;
					double c2 = c;
					double c3 = c;
					double el1 = ie[l + 1];
					double s = 0.0;
					double s2 = 0.0;
					for (int i = m - 1; i >= l; i--) {
						c3 = c2;
						c2 = c;
						s2 = s;
						g = c * ie[i];
						h = c * p;
						r = Math.hypot(p, ie[i]);
						ie[i + 1] = s * r;
						s = ie[i] / r;
						c = p / r;
						p = c * re[i] - s * g;
						re[i + 1] = h + s * (c * g + s * re[i]);
	
						// Accumulate transformation.
						for (int k = 0; k < n; k++) {
							h = V[k][i + 1];
							V[k][i + 1] = s * V[k][i] + c * h;
							V[k][i] = c * V[k][i] - s * h;
						}
					}
					p = -s * s2 * c3 * el1 * ie[l] / dl1;
					ie[l] = s * p;
					re[l] = c * p;
	
					// Check for convergence.
				} while (Math.abs(ie[l]) > ArrUtils.EPSILON * tst1);
			}
			re[l] = re[l] + f;
			ie[l] = 0.0;
		}
	
		// Sort eigenvalues and corresponding vectors.
		for (int i = 0; i < n - 1; i++) {
			// find max value in [i:n]
			int k = i;
			double val = re[i];
			for (int j = i + 1; j < n; j++) {
				if (re[j] > val) {
					k = j; // store index of max value
					val = re[j];
				}
			}
			if (k != i) {
				// put max value at i
				re[k] = re[i];
				re[i] = val;
				for (int j = 0; j < n; j++) {
					val = V[j][i];
					V[j][i] = V[j][k];
					V[j][k] = val;
				}
			}
		}
	}
}
