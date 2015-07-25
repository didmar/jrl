package com.github.didmar.jrl.environment;

import com.github.didmar.jrl.environment.Environment;
import com.github.didmar.jrl.features.RBFFeatures;
import com.github.didmar.jrl.utils.array.ArrUtils;

// TODO à transformer en DynSysEnv
/**
 * A benchmark environment with continuous states and actions and randomly
 * generated reward and dynamics. The reward function and the dynamics are
 * linear combinations of {@link RBFFeatures}. All state and action dimensions
 * are bounded in [0,1].
 * @author Didier Marin
 */
public class NonLinearRandomizedEnvironment extends Environment {

	/** Features to code the reward function */
	private final RBFFeatures rewFeatures;
	/** Weights to code the reward function */
	private final double[] rewWeigths;
	/** Features to code the dynamics */
	private RBFFeatures dynFeatures;
	/** Weights to code the dynamics */
	private final double[][] dynWeigths;
	/** Time step of the simulation */
	private final double dt;
	/** State-space lower bound */ 
	private final double[] xMin;
	/** State-space upper bound */
	private final double[] xMax;
	
	private final double[] xu;
	private final double[] phixuRew;
	private final double[] phixuDyn;
	private final double[] xn;
	
	/**
	 * @param xDim
	 * @param uDim
	 */
	public NonLinearRandomizedEnvironment(int xDim, int uDim, int nbRewFeatures,
			int nbDynFeatures, double dt) {
		super(xDim, uDim);
		this.dt = dt;
		xMin = ArrUtils.zeros(xDim);
		xMax = ArrUtils.ones(xDim);
		// Generate the reward function
		double[][] rewCenters = new double[nbRewFeatures][xDim+uDim];
		for (int i = 0; i < nbRewFeatures; i++) {
			for (int j = 0; j < xDim+uDim; j++) {
				rewCenters[i][j] = Math.random();
			}
		}
		double[] rewSigma = ArrUtils.constvec(xDim+uDim, 0.01);
		rewFeatures = new RBFFeatures(rewCenters, rewSigma, true);
		rewWeigths = ArrUtils.rand(rewFeatures.outDim);
		// Generate the dynamics
		double[][] dynCenters = new double[nbDynFeatures][xDim+uDim];
		for (int i = 0; i < nbRewFeatures; i++) {
			for (int j = 0; j < xDim+uDim; j++) {
				dynCenters[i][j] = Math.random();
			}
		}
		double[] dynSigma = ArrUtils.constvec(xDim+uDim, 0.01);
		dynFeatures = new RBFFeatures(dynCenters, dynSigma, true);
		dynWeigths = new double[xDim][dynFeatures.outDim];
		for (int i = 0; i < xDim; i++) {
			ArrUtils.rand(dynWeigths[i]);
		}
		
		xu = new double[xDim+uDim];
		phixuRew = new double[rewFeatures.outDim];
		phixuDyn = new double[dynFeatures.outDim];
		xn = new double[xDim];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#isTerminal(double[], double[], double[])
	 */
	@Override
	public boolean isTerminal(double[] x, double[] u, double[] xn) {
		return false;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#nextState(double[], double[])
	 */
	@Override
	public double[] nextState(double[] x, double[] u) {
		System.arraycopy(x, 0, xu, 0,    xDim);
		System.arraycopy(u, 0, xu, xDim, uDim);
		dynFeatures.phi(xu, phixuDyn);
		for (int i = 0; i < xDim; i++) {
			// TODO add a noise term
			xn[i] = x[i] + dt*ArrUtils.dotProduct(phixuDyn, dynWeigths[i], phixuDyn.length);
		}
		ArrUtils.boundVector(xn,xMin,xMax);
		return xn;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#reward(double[], double[], double[])
	 */
	@Override
	public double reward(double[] x, double[] u, double[] xn) {
		System.arraycopy(x, 0, xu, 0,    xDim);
		System.arraycopy(u, 0, xu, xDim, uDim);
		rewFeatures.phi(xu,phixuRew);
		return ArrUtils.dotProduct(phixuRew, rewWeigths, phixuRew.length);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	public double[] startState() {
		return ArrUtils.constvec(xDim, 0.5);
	}
	
	@Override
	public final double[] getXMin() {
		return xMin;
	}

	@Override
	public final double[] getXMax() {
		return xMax;
	}
}
