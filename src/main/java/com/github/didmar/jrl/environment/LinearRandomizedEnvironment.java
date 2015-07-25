package com.github.didmar.jrl.environment;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.environment.Environment;
import com.github.didmar.jrl.features.RBFFeatures;
import com.github.didmar.jrl.utils.array.ArrUtils;

// TODO make LinearRandomizedEnvironment a DynSysEnvironement
/**
 * A benchmark environment with continuous states and actions and randomly
 * generated reward and dynamics. The reward function is a linear combination
 * of {@link RBFFeatures} and the dynamics is linear combination in the state
 * and the action. All state and action dimensions are bounded in [0,1].
 * @author Didier Marin
 */
public class LinearRandomizedEnvironment extends Environment {

	/** Features to code the reward function */
	private final RBFFeatures rewFeatures;
	/** Weights to code the reward function */
	private final double[] rewWeights;
	/** Used to code the dynamics */
	private final double[][] A;
	/** Used to code the dynamics */
	private final double[][] B;
	/** Used to code the dynamics */
	private final double[] C;
	/** Used to code the dynamics */
	private final double[] D;
	/** Time step of the simulation */
	private final double dt;
	/** State-space lower bound */
	private final double[] xMin;
	/** State-space upper bound */
	private final double[] xMax;

	private final double[] xu;
	private final double[] phixu;
	private final double[] xnStorage;
	private final double[] tmp1;
	private final double[] tmp2;

	public LinearRandomizedEnvironment(int xDim, int uDim, int nbRewFeatures,
			double dt) {
		super(xDim, uDim);
		this.dt = dt;
		xMin = ArrUtils.zeros(xDim);
		xMax = ArrUtils.ones(xDim);
		// Generate the reward function
		double[][] centers = new double[nbRewFeatures][xDim+uDim];
		for (int i = 0; i < nbRewFeatures; i++) {
			for (int j = 0; j < xDim+uDim; j++) {
				centers[i][j] = Math.random();
			}
		}
		double[] sigma = ArrUtils.constvec(xDim+uDim, 0.01);
		rewFeatures = new RBFFeatures(centers, sigma, true);
		rewWeights = ArrUtils.rand(rewFeatures.outDim);
		// Generate the dynamics
		A = new double[xDim][xDim];
		for (int i = 0; i < xDim; i++) {
			for (int j = 0; j < xDim; j++) {
				A[i][j] = Math.random()-0.5; // in [-0.5,+0.5]
			}
		}
		B = new double[xDim][uDim];
		for (int i = 0; i < xDim; i++) {
			for (int j = 0; j < uDim; j++) {
				B[i][j] = Math.random()-0.5; // in [-0.5,+0.5]
			}
		}
		C = new double[xDim];
		for (int i = 0; i < xDim; i++) {
			C[i] = Math.random()-0.5;
		}
		D = new double[xDim];
		for (int i = 0; i < xDim; i++) {
			D[i] = 10.*Math.random();
		}

		xu = new double[xDim+uDim];
		phixu = new double[rewFeatures.outDim];
		xnStorage = new double[xDim];
		tmp1 = new double[xDim];
		tmp2 = new double[xDim];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#isTerminal(double[], double[], double[])
	 */
	@Override
	public final boolean isTerminal(@NonNull double[] x,
									@NonNull double[] u,
									@NonNull double[] xn) {
		return false;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#nextState(double[], double[])
	 */
	@Override
	@NonNull
	public final double[] nextState(@NonNull double[] x,
									@NonNull double[] u) {
		assert x.length == xDim;
		assert u.length == uDim;

		ArrUtils.multiply(A, x, tmp1, xDim, xDim);
		ArrUtils.multiply(B, u, tmp2, xDim, uDim);
		for (int i = 0; i < xDim; i++) {
			// TODO add a noise term
			xnStorage[i] = x[i] + dt*(tmp1[i] + tmp2[i] + C[i]*Math.sin(D[i]*(tmp1[i]+tmp2[i])));
		}
		ArrUtils.boundVector(xnStorage,xMin,xMax);
		return xnStorage;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#reward(double[], double[], double[])
	 */
	@Override
	public final double reward(@NonNull double[] x,
							   @NonNull double[] u,
							   @NonNull double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;

		System.arraycopy(x, 0, xu, 0,    xDim);
		System.arraycopy(u, 0, xu, xDim, uDim);
		rewFeatures.phi(xu,phixu);
		return ArrUtils.dotProduct(phixu, rewWeights, phixu.length);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	@NonNull
	public final double[] startState() {
		return ArrUtils.constvec(xDim, 0.5);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getXMin()
	 */
	@Override
	@NonNull
	public final double[] getXMin() {
		return xMin;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getXMax()
	 */
	@Override
	@NonNull
	public final double[] getXMax() {
		return xMax;
	}
}
