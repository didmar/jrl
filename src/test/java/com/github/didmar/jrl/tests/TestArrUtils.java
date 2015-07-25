package com.github.didmar.jrl.tests;

import static org.junit.Assert.*;

import org.eclipse.jdt.annotation.NonNull;
import org.eclipse.jdt.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.Utils;
import com.github.didmar.jrl.utils.array.ArrUtils;

import Jama.Matrix;

/**
 * Unit test class for {@link Utils}
 * @author Didier Marin
 */
public class TestArrUtils {

	static final int SIZE_BIGVEC = 100;
	private final double[] vec1 = {1.0, 5.0, 0., -1.0};
	private final double[] vec2 = {1.0, 5.0, 0., 0.0};
	private final double[][] vecBounds = {{0., 2.},{6.,10.},{0.,-1.},{0.,0.}};
	private final double[] bigvec = new double[SIZE_BIGVEC];
	private final double[][] mat = new double[][]
			{{ 0.47054337, -0.36687449,  0.80307445,  2.36231204},
	         {-0.03714442,  0.34631978, -0.11733385,  0.68553154},
	         { 0.13961631, -1.74065795, -0.09071681, -0.29429432},
	         { 0.31820711,  0.45183428, -0.71429973, -1.25328148}};
	
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
	 * Test method for {@link jrl.utils.ArrUtils#sum(double[])}.
	 */
	@Test
	public void testSum() {
		assertTrue(ArrUtils.sum(vec1) == 5.0);
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#mean(double[])}.
	 */
	@Test
	public void testMean() {
		assertTrue(ArrUtils.mean(vec1) == 1.25);
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#var(double[])}.
	 */
	@Test
	public void testVar() {
		assertTrue(Math.abs( ArrUtils.var(vec1) - 6.916666666666667) < 1e-5);
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#std(double[])}.
	 */
	@Test
	public void testStd() {
		assertTrue(Math.abs( ArrUtils.std(vec1) - Math.sqrt(6.916666666666667)) < 1e-5);
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#resizeArray(java.lang.Object, int)}.
	 */
	@Test
	public void testResizeArray() {
		final int size = vec1.length;
		final int newSize = size + 5;
		final double[] newVec = (double[]) ArrUtils.resizeArray(vec1, newSize);
		assertTrue(newVec.length == newSize);
		for(int i=0; i<size; i++) {
			assertTrue(vec1[i] == newVec[i]);
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#arrayEquals(double[], double[])}.
	 */
	@Test
	public void testArrayEquals() {
		assertTrue(ArrUtils.arrayEquals(vec1,vec1));
		assertFalse(ArrUtils.arrayEquals(vec1,vec2));
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#constvec(int, double)}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testConstvecIntDouble() {
		final int l = 3;
		final double cst = 0.123;
		final double[] cstvec = ArrUtils.constvec(l, cst);
		assertTrue(cstvec.length == l);
		for(int i=0; i<l; i++) {
			assertTrue(cstvec[i] == cst);
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#ones(int)}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testOnesInt() {
		final int l = 3;
		final double[] onesvec = ArrUtils.ones(l);
		assertTrue(onesvec.length == l);
		for(int i=0; i<l; i++) {
			assertTrue(onesvec[i] == 1.);
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#zeros(int)}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testZerosInt() {
		final int l = 3;
		final double[] zerosvec = ArrUtils.zeros(l);
		assertTrue(zerosvec.length == l);
		for(int i=0; i<l; i++) {
			assertTrue(zerosvec[i] == 0.);
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#constvec(double[], double)}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testConstvecDoubleArrayDouble() {
		final int l = 3;
		final double cst = 0.123;
		final double[] cstvec = new double[l];
		ArrUtils.constvec(cstvec, cst);
		assertTrue(cstvec.length == l);
		for(int i=0; i<l; i++) {
			assertTrue(cstvec[i] == cst);
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#ones(double[])}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testOnesDoubleArray() {
		final int l = 3;
		final double[] onesvec = new double[l];
		ArrUtils.ones(onesvec);
		assertTrue(onesvec.length == l);
		for(int i=0; i<l; i++) {
			assertTrue(onesvec[i] == 1.);
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#zeros(double[])}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testZerosDoubleArray() {
		final int l = 3;
		final double[] zerosvec = new double[l];
		ArrUtils.zeros(zerosvec);
		assertTrue(zerosvec.length == l);
		for(int i=0; i<l; i++) {
			assertTrue(zerosvec[i] == 0.);
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#constmat(double[][], double)}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testConstmat() {
		final int l1 = 3;
		final int l2 = 4;
		final double cst = 0.123;
		final double[][] cstmat = new double[l1][l2];  
		ArrUtils.constmat(cstmat, cst);
		assertTrue(cstmat.length == l1 && cstmat[0].length == l2);
		for(int i=0; i<l1; i++) {
			for(int j=0; j<l2; j++) {
				assertTrue(cstmat[i][j] == cst);
			}
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#ones(double[][])}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testOnesDoubleArrayArray() {
		final int l1 = 3;
		final int l2 = 4;
		final double[][] onesmat = new double[l1][l2];  
		ArrUtils.ones(onesmat);
		assertTrue(onesmat.length == l1 && onesmat[0].length == l2);
		for(int i=0; i<l1; i++) {
			for(int j=0; j<l2; j++) {
				assertTrue(onesmat[i][j] == 1.);
			}
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#zeros(double[][])}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testZerosDoubleArrayArray() {
		final int l1 = 3;
		final int l2 = 4;
		final double[][] zerosmat = new double[l1][l2];  
		ArrUtils.zeros(zerosmat);
		assertTrue(zerosmat.length == l1 && zerosmat[0].length == l2);
		for(int i=0; i<l1; i++) {
			for(int j=0; j<l2; j++) {
				assertTrue(zerosmat[i][j] == 0.);
			}
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#norm(double[])}.
	 */
	@Test
	public void testNorm() {
		assertTrue(Math.abs( ArrUtils.norm(vec1) - 5.196152422706632) < 1e-5);
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#squaredNorm(double[])}.
	 */
	@Test
	public void testSquaredNorm() {
		assertTrue(Math.abs( ArrUtils.squaredNorm(vec1)
				- Math.pow(5.196152422706632,2)) < 1e-5);
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#boundVector(double[], double[][])}.
	 */
	@Test
	public void testBoundVector() {
		ArrUtils.boundVector(vec1, vecBounds);
		assertTrue(vec1[0]==1.);
		assertTrue(vec1[1]==6.);
		assertTrue(vec1[2]==-1.);
		assertTrue(vec1[3]==0.);
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#linspace(double, double, int)}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testLinspace() {
		final double[] lin = ArrUtils.linspace(0.,1.,5);
		assertTrue(lin[0]==0.);
		assertTrue(lin[1]==0.25);
		assertTrue(lin[2]==0.5);
		assertTrue(lin[3]==0.75);
		assertTrue(lin[4]==1.0);
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#buildGrid(double[], double[], int[])}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testBuildGridDoubleArrayDoubleArrayIntArray() {
		@NonNull final double[] mins = {0.,-1.,10.,0.};
		@NonNull final double[] maxs = {1., 1.,15.,1.};
		@NonNull final int[] steps= {3,  2,  2,  1};
		@NonNull final double[][] grid = ArrUtils.buildGrid(mins, maxs, steps);
		assertTrue(grid.length == 3*2*2*1 && grid[0].length == 4);
		@NonNull final double[][] goodGrid =
		{{0.0,-1.0,10.0,0.0},{0.0,-1.0,15.0,0.0},{0.0,1.0,10.0,0.0},{0.0,1.0,15.0,0.0},
		 {0.5,-1.0,10.0,0.0},{0.5,-1.0,15.0,0.0},{0.5,1.0,10.0,0.0},{0.5,1.0,15.0,0.0},
		 {1.0,-1.0,10.0,0.0},{1.0,-1.0,15.0,0.0},{1.0,1.0,10.0,0.0},{1.0,1.0,15.0,0.0}};
		for(int i=0; i<grid.length; i++) {
			assertTrue(ArrUtils.arrayEquals(grid[i], goodGrid[i]));
		}
	}

	/**
	 * Test method for {@link jrl.utils.ArrUtils#buildGrid(double[], double[], int)}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testBuildGridDoubleArrayDoubleArrayInt() {
		@NonNull final double[] mins = {0.,2.};
		@NonNull final double[] maxs = {1.,3.};
		final int step = 3;
		@NonNull final double[][] grid = ArrUtils.buildGrid(mins, maxs, step);
		assertTrue(grid.length == 3*3 && grid[0].length == 2);
		@NonNull final double[][] goodGrid =
			{{0.0,2.0},{0.0,2.5},{0.0,3.0},
			 {0.5,2.0},{0.5,2.5},{0.5,3.0},
			 {1.0,2.0},{1.0,2.5},{1.0,3.0}};
		for(int i=0; i<grid.length; i++) {
			assertTrue(ArrUtils.arrayEquals(grid[i], goodGrid[i]));
		}
	}
	
	/**
	 * Test method for {@link jrl.utils.ArrUtils#isSymmetric(double[][])}.
	 */
	@Test
	public void testIsSymmetric() {
		final int n = vec1.length;
		@NonNull final double[][] A = new double[n][n];
		ArrUtils.multiply(vec1, vec1, A, n);
		assertTrue(ArrUtils.isSymmetric(A));
		
		ArrUtils.multiply(vec1, vec2, A, n);
		assertFalse(ArrUtils.isSymmetric(A));
	}
	
	/**
	 * Test method for {@link jrl.utils.ArrUtils#squareRootInPlace(double[][], int)}.
	 */
	@Test
	public void testSquareRootInPlace() {
		
		// Symmetric matrix case
		
		int n = vec1.length;
		@NonNull final double[][] A = ArrUtils.multiply(vec1, vec1, n);
		
		@NonNull final double[][] R = new double[n][n];
		ArrUtils.copyMatrix(A, R, n);
		try {
			ArrUtils.squareRootInPlace(R, n);
		} catch (Exception e) {
			assertFalse(true);
		}
		
		@NonNull final double[][] RR = new double[n][n];
		ArrUtils.multiplyQuad(R, R, RR, n);
		
		assertTrue( ArrUtils.allClose(A, RR, Utils.getMacheps()) );
		
		//TODO Test non-symmetric matrix case (not implemented yet !)
//		int m = 3;
//		double[][] B = new double[][]{{2,1,0},{3,2,0},{0,0,4}};
//		assert !Utils.isSymmetric(B);
//		
//		double[][] R2 = new double[m][m];
//		Utils.copyMatrix(B, R2, m);
//		Utils.squareRootInPlace(R2, m);
//		
//		double[][] RR2 = new double[m][m];
//		Utils.multiplyQuad(R2, R2, RR2, m);
//		
//		System.out.println("B="+Utils.toString(B));
//		System.out.println("RR2="+Utils.toString(RR2));
//		
//		assertTrue( Utils.allClose(B, RR2, Utils.getMacheps()) );
	}
	
	/**
	 * Test method for {@link jrl.utils.ArrUtils#pinv(Matrix)}.
	 */
	@SuppressWarnings("static-method")
	@Test
	public void testPinvMatrix() {
		@NonNull Matrix A = new Matrix(new double[][]{{2,2,3},{6,6,9},{1,4,8}});
		@Nullable double[][] Ainv = null;
		try {
			Ainv = ArrUtils.pinv(A).getArray();
		} catch (Exception e) {
			// No need to do anything
		}
		assertTrue(Ainv != null);
		
		// TODO find a "true" with more precision
		@NonNull final double[][] trueAinv = new double[][]{
				{0.0579,0.1738,-0.2308},
				{0.0118,0.0353,-0.0000},
				{-0.0131,-0.0394, 0.1538}};
		assertTrue( ArrUtils.allClose(Ainv, trueAinv, 1e-4) );
		
	}
	
	/**
	 * Test method for {@link jrl.utils.Utils#shermanMorrisonFormula(double[][],
	 * 		double[] u,	double[], int)}.
	 */
	@Test
	public void testShermanMorrisonFormula() {
		@NonNull final double[][] vec1vec2 = ArrUtils.multiply(vec1, vec2, vec1.length);
		@NonNull final double[][] newmat = ArrUtils.emptyLike(mat);
		for (int i = 0; i < newmat.length; i++) {
			for (int j = 0; j < newmat[i].length; j++) {
				newmat[i][j] = mat[i][j] + vec1vec2[i][j];
			}
		}
		@Nullable double[][] newmatinv = null;
		try {
			newmatinv = ArrUtils.pinv(new Matrix(newmat)).getArray();
		} catch (Exception e1) {
			assertTrue(false);
		}
		@Nullable double[][] Ainv = null;
		try {
			Ainv = ArrUtils.pinv(new Matrix(mat)).getArray();
		} catch (Exception e1) {
			assertTrue(false);
		}
		boolean exceptionCatched = false;
		if(Ainv != null && newmatinv != null) {
			try {
				ArrUtils.shermanMorrisonFormula(Ainv, vec1, vec2, vec1.length);
			} catch (Exception e) {
				exceptionCatched = true;
			}
			assertFalse(exceptionCatched);
			assertTrue(ArrUtils.allClose(Ainv, newmatinv, 1e-10));
		}
	}
}
