package com.github.didmar.jrl.environment.discrete;

import com.github.didmar.jrl.environment.Environment;

/**
 * Environment with discrete one-dimensional states and actions.
 * @author Didier Marin
 */
public abstract class DiscreteEnvironment extends Environment {

	/** State-space cardinality : states are in [0,xCard[ */
	public final int xCard;
	/** Action-space cardinality : actions are in [0,uCard[ */
	public final int uCard;
	/** State-space lower bound */
	private final double[] xMin;
	/** State-space upper bound */
	private final double[] xMax;
	/** Action-space lower bound */
	private final double[] uMin;
	/** Action-space upper bound */
	private final double[] uMax;

	public DiscreteEnvironment(int xCard, int uCard) {
		super(1, 1);
		if(xCard <= 0) {
			throw new IllegalArgumentException("xCard must be greater than 0");
		}
		if(uCard <= 0) {
			throw new IllegalArgumentException("uCard must be greater than 0");
		}
		this.xCard = xCard;
		this.uCard = uCard;
		xMin = new double[]{0.};
		xMax = new double[]{xCard-1.};
		uMin = new double[]{0.};
		uMax = new double[]{uCard-1.};
	}

	/**
	 * Returns the state-space cardinality.
	 * @return the state-space cardinality
	 */
	public final int getXCard() {
		return xCard;
	}

	/**
	 * Returns the action-space cardinality.
	 * @return the action-space cardinality
	 */
	public final int getUCard() {
		return uCard;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getXMin()
	 */
	@Override
	public final double[] getXMin() {
		return xMin;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getXMax()
	 */
	@Override
	public final double[] getXMax() {
		return xMax;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getUMin()
	 */
	@Override
	public final double[] getUMin() {
		return uMin;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getUMax()
	 */
	@Override
	public final double[] getUMax() {
		return uMax;
	}
}
