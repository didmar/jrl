package com.github.didmar.jrl.environment.discrete;

import com.github.didmar.jrl.environment.discrete.DiscreteEnvironment;
import com.github.didmar.jrl.mdp.DiscreteMDP;
import com.github.didmar.jrl.utils.RandUtils;

// TODO make the connection with DiscreteMDP more clear
/**
 * An environment based on a discrete one-dimensional state and action MDP with
 * an initial state distribution P0, a transition function P and deterministic
 * reward function R.
 * @author Didier Marin
 */
public class DiscreteMDPEnvironment extends DiscreteEnvironment {

	/** Underlying MDP */
	protected final DiscreteMDP mdp;

	/**
	 * Construct a DiscreteEnvironment given an MDP.
	 * @param MDP  the MDP
	 */
	public DiscreteMDPEnvironment(DiscreteMDP mdp) {
		super(mdp.P0.length, mdp.P[0].length);
		this.mdp = mdp;
	}

	/**
	 * Construct a DiscreteEnvironment given the initial state distribution,
	 * the transition and the reward function.
	 * @param P0   initial state probability table
	 * @param P    transition probability table
	 * @param R    reward table
	 */
	public DiscreteMDPEnvironment(double[] P0, double[][][] P,	double[][] R) {
		super(P0.length, P[0].length);
		mdp = new DiscreteMDP(P0, P, R);
	}

//	/**
//	 * Construct a DiscreteEnvironment given the state and action cardinality.
//	 * P, P0 and R are initialized according to the cardinality, and should be
//	 * filled with the proper values.
//	 * @param xCard    state-space cardinality
//	 * @param uCard    action-space cardinality
//	 */
//	public DiscreteMDPEnvironment(int xCard, int uCard) {
//		super(xCard,uCard);
//		this.P0 = new double[xCard];
//		this.P = new double[xCard][uCard][xCard];
//		this.R = new double[xCard][uCard];
//	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#nextState(double[], double[])
	 */
	@Override
	public final double[] nextState(double[] x, double[] u) {
		if(x==null) throw new IllegalArgumentException("x must not be null");
		if(u==null) throw new IllegalArgumentException("u must not be null");
		if(x.length != xDim ){
			throw new IllegalArgumentException("x must have length xDim");
		}
		if(u.length != uDim ){
			throw new IllegalArgumentException("u must have length uDim");
		}

		return new double[]{RandUtils.drawFromDiscreteProbTable(mdp.P[(int) x[0]][(int) u[0]])};
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#reward(double[], double[], double[])
	 */
	@Override
	public final double reward(double[] x, double[] u, double[] xn) {
		if(x==null) throw new IllegalArgumentException("x must not be null");
		if(u==null) throw new IllegalArgumentException("u must not be null");
		if(x.length != xDim ){
			throw new IllegalArgumentException("x must have length xDim");
		}
		if(u.length != uDim ){
			throw new IllegalArgumentException("u must have length uDim");
		}

		return mdp.R[(int)x[0]][(int)u[0]];
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	public final double[] startState() {
		return new double[]{RandUtils.drawFromDiscreteProbTable(mdp.P0)};
	}

	public final double[] getP0() {
		return mdp.P0;
	}

	public final double[][][] getP() {
		return mdp.P;
	}

	public final double[][] getR() {
		return mdp.R;
	}

	/**
	 * No terminal sample, by default.
	 */
	@Override
	public boolean isTerminal(double[] x, double[] u, double[] xn) {
		return false;
	}
}
