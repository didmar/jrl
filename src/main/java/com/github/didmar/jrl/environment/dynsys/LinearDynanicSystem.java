package com.github.didmar.jrl.environment.dynsys;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * A {@link DynamicSystem} with linear dynamics: dotX = A*x + B*u
 * @author Didier Marin
 */
public final class LinearDynanicSystem implements DynamicSystem {

	private final double[][] A;
	private final double[][] B;

	public LinearDynanicSystem(double[][] A, double[][] B) {
		if(!ArrUtils.isQuadratic(A)) {
			throw new IllegalArgumentException("A must be a square matrix");
		}
		if(A.length != B.length) {
			throw new IllegalArgumentException("A and B must have the same number of lines");
		}
		this.A = A;
		this.B = B;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.dynsys.DynamicalSystem#dotX(double[], double[], double[])
	 */
	public final void dotX(@NonNull final double[] x, @NonNull final double[] u,
						   @NonNull final double[] dotX) {
		for (int i = 0; i < dotX.length; i++) {
			dotX[i] = 0.;
			for (int j = 0; j < A[i].length; j++) {
				dotX[i] += A[i][j] * x[j];
			}
			for (int j = 0; j < B[i].length; j++) {
				dotX[i] += B[i][j] * u[j];
			}
		}
	}

	/* (non-Javadoc)
	 * @see jrl.environment.dynsys.DynamicalSystem#getXDim()
	 */
	public final int getXDim() {
		return A.length;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.dynsys.DynamicalSystem#getUDim()
	 */
	public final int getUDim() {
		return B[0].length;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.dynsys.DynamicSystem#derX(double[], double[], double[][])
	 */
	public void derX(@NonNull final double[] x, @NonNull final double[] u,
					 @NonNull final double[][] derX) {
		assert ArrUtils.hasShape(derX, A.length, A[0].length);
		
		ArrUtils.copyMatrix(A, derX, A.length, A[0].length);
		
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.dynsys.DynamicSystem#derU(double[], double[], double[][])
	 */
	public void derU(@NonNull final double[] x, @NonNull final double[] u,
					 @NonNull final double[][] derU) {
		assert ArrUtils.hasShape(derU, B.length, B[0].length);
		
		ArrUtils.copyMatrix(B, derU, B.length, B[0].length);
	}

}
