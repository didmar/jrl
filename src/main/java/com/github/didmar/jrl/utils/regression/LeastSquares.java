package com.github.didmar.jrl.utils.regression;

import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.utils.ParametricFunction;
import com.github.didmar.jrl.utils.array.ArrUtils;

import Jama.Matrix;

/**
 * Linear regression with Least Squares.
 * @author Didier Marin
 */
public final class LeastSquares implements ParametricFunction {

	private final int n;
	private final double[] params;
	private final Features feat;
	private final Matrix A;
	private final double[] b;
	/** Regularization term, ignored if 0 */
	private final double reg;

	// used for temporary storage
	private final double[] phiX;
	private final double[][] tmp;

	public LeastSquares(Features feat, double reg) {
		if(feat == null) {
			throw new IllegalArgumentException("feat must be non-null");
		}
		this.feat = feat;
		n = feat.outDim;
		params = ArrUtils.zeros(n);
		A = new Matrix( ArrUtils.zeros(n,n) );
		b = ArrUtils.zeros(n);
		this.reg = reg;

		phiX = new double[n];
		tmp = new double[n][n];
	}

	public final void addSample(double[] x, double y) {
		assert x != null;
		assert x.length == feat.inDim;

		// Compute the features for the given input
		feat.phi(x,phiX);
		ArrUtils.multiply(phiX, phiX, tmp, n);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				A.set(i, j, A.get(i,j) + tmp[i][j]);
			}
			b[i] += phiX[i] * y;
		}
	}

	public final void computeParameters() {
		Matrix AplusReg = A;
		// Use regularization
		if(reg > 0.) {
			AplusReg = A.copy();
			for (int i = 0; i < n; i++) {
				AplusReg.set(i,i, AplusReg.get(i,i)+reg);
			}
		}
		double[][] Ainv = null;
		try {
			Ainv = ArrUtils.pinv(AplusReg).getArray();
		} catch (Exception e) {
			// FIXME handle this exception properly
			throw new RuntimeException("Could not compute pseuso-inverse of A");
		}
		ArrUtils.multiply(Ainv, b, params, n, n);
	}

	public final double predict(double[] x) {
		assert x != null;
		assert x.length == feat.inDim;

		// Compute the features for the given input
		feat.phi(x,phiX);
		// Compute the predicted output
		return ArrUtils.dotProduct(phiX, params, n);
	}

	@Override
	public final boolean boundParams(double[] params) {
		return true;
	}

	@Override
	public final double[] getParams() {
		return params;
	}

	@Override
	public final int getParamsSize() {
		return n;
	}

	@Override
	public final void setParams(double[] params) {
		if(params.length != this.params.length) {
			throw new IllegalArgumentException("Incorrect parameters length");
		}
		System.arraycopy(params, 0, this.params, 0, n);
	}

	@Override
	public final void updateParams(double[] delta) {
		throw new UnsupportedOperationException();
	}

	public final Features getFeatures() {
		return feat;
	}
}
