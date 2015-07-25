package com.github.didmar.jrl.utils.array;

/**
 * LU Decomposition.
 * <P>
 * For an n-by-n matrix A, the LU decomposition is an n-by-n unit lower
 * triangular matrix L, an n-by-n upper triangular matrix U, and a
 * permutation vector piv of length n so that A(piv,:) = L*U.
 * <P>
 * The LU decompostion with pivoting always exists, even if the matrix is
 * singular, so the constructor will never fail. The primary use of the LU
 * decomposition is in the solution of square systems of simultaneous linear
 * equations. This will fail if isNonsingular() returns false.
 */
final class LUDecomposition {

	/**
	 * Array for internal storage of decomposition.
	 * 
	 * @serial internal array storage.
	 */
	private final double[][] LU;

	/**
	 * Row and column dimensions, and pivot sign.
	 * 
	 * @serial column dimension.
	 * @serial row dimension.
	 * @serial pivot sign.
	 */
	private final int n;

	/**
	 * Internal storage of pivot vector.
	 * 
	 * @serial pivot vector.
	 */
	private final int[] piv;

	/**
	 * LU Decomposition
	 * 
	 * @param A
	 *            quadratic matrix
	 * @param n
	 *            size of A
	 */
	LUDecomposition(final double[][] A, int n) {
		assert ArrUtils.hasShape(A, n, n);
		
		// Use a "left-looking", dot-product, Crout/Doolittle algorithm.
		this.n = n;
		LU = new double[n][n];
		for (int i = 0; i < n; i++) {
			System.arraycopy(A[i], 0, LU[i], 0, n);
		}
		piv = new int[n];
		for (int i = 0; i < n; i++) {
			piv[i] = i;
		}
		int pivsign = 1;
		double[] LUrowi;
		double[] LUcolj = new double[n];
		// Outer loop.
		for (int j = 0; j < n; j++) {
			// Make a copy of the j-th column to localize references.
			for (int i = 0; i < n; i++) {
				LUcolj[i] = LU[i][j];
			}
			// Apply previous transformations.
			for (int i = 0; i < n; i++) {
				LUrowi = LU[i];
				// Most of the time is spent in the following dot product.
				int kmax = Math.min(i, j);
				double s = 0.0;
				for (int k = 0; k < kmax; k++) {
					s += LUrowi[k] * LUcolj[k];
				}
				LUrowi[j] = LUcolj[i] -= s;
			}
			// Find pivot and exchange if necessary.
			int p = j;
			for (int i = j + 1; i < n; i++) {
				if (Math.abs(LUcolj[i]) > Math.abs(LUcolj[p])) {
					p = i;
				}
			}
			if (p != j) {
				for (int k = 0; k < n; k++) {
					double t = LU[p][k];
					LU[p][k] = LU[j][k];
					LU[j][k] = t;
				}
				int k = piv[p];
				piv[p] = piv[j];
				piv[j] = k;
				pivsign = -pivsign;
			}
			// Compute multipliers.
			if (j < n & LU[j][j] != 0.0) {
				for (int i = j + 1; i < n; i++) {
					LU[i][j] /= LU[j][j];
				}
			}
		}
	}

	/**
	 * Solve A*X = B
	 * 
	 * @param B
	 *            A Matrix with as many rows as A and any number of columns.
	 * @return X so that L*U*X = B(piv,:)
	 * @exception IllegalArgumentException
	 *                Matrix row dimensions must agree.
	 * @exception RuntimeException
	 *                Matrix is singular.
	 */
	public final double[][] solve(final double[][] B) {
		assert B != null;
		assert B.length != n;
		
		if (!isNonsingular()) {
			throw new IllegalStateException("Matrix is singular.");
		}
		// Copy right hand side with pivoting
		int nx = B[0].length;
		double[][] X = getMatrix(B, piv, 0, nx - 1);
		// Solve L*Y = B(piv,:)
		for (int k = 0; k < n; k++) {
			for (int i = k + 1; i < n; i++) {
				for (int j = 0; j < nx; j++) {
					X[i][j] -= X[k][j] * LU[i][k];
				}
			}
		}
		// Solve U*X = Y;
		for (int k = n - 1; k >= 0; k--) {
			for (int j = 0; j < nx; j++) {
				X[k][j] /= LU[k][k];
			}
			for (int i = 0; i < k; i++) {
				for (int j = 0; j < nx; j++) {
					X[i][j] -= X[k][j] * LU[i][k];
				}
			}
		}
		return X;
	}

	/**
	 * Is the matrix nonsingular?
	 * 
	 * @return true if U, and hence A, is nonsingular.
	 */
	private boolean isNonsingular() {
		for (int j = 0; j < n; j++) {
			if (LU[j][j] == 0) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Get a submatrix.
	 * 
	 * @param A
	 *            matrix
	 * @param r
	 *            Array of row indices.
	 * @param j0
	 *            Initial column index
	 * @param j1
	 *            Final column index
	 * @return A(r(:),j0:j1)
	 * @exception ArrayIndexOutOfBoundsException
	 *                Submatrix indices
	 */
	private static double[][] getMatrix(final double[][] A,
									    final int[] r,
										int j0, int j1) {
		assert A != null;
		assert r != null;
		assert j0 > 0;
		assert j0 < j1;
		
		double[][] B = new double[r.length][j1 - j0 + 1];
		for (int i = 0; i < r.length; i++) {
			assert r[i] > 0 && r[i] < A.length;
			assert j1 < A[r[i]].length;
			for (int j = j0; j <= j1; j++) {
				B[i][j - j0] = A[r[i]][j];
			}
		}
		return B;
	}
}