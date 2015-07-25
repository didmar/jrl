package com.github.didmar.jrl.environment;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.environment.dynsys.DynSysEnvironment;
import com.github.didmar.jrl.environment.dynsys.LinearDynanicSystem;
import com.github.didmar.jrl.utils.RandUtils;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Reaching task with a point mass, no dynamics (i.e., xn = x + u).
 * 
 * @author Didier Marin
 */
public final class PointMass extends DynSysEnvironment implements ReachingTask {
	
	private static LinearDynanicSystem linDynSys
		= new LinearDynanicSystem(new double[][]{{0.}},new double[][]{{1.}});
	
	public enum PointMassRewardType {
		COST, NORMALIZED_REWARD, GOAL_REWARD;
	}
	
	/** Radius of the target position, for the GOAL_REWARD task */
	private final double goalRadius = 0.01;
	/** Action-space lower bound */
	private static final double[] uMin = new double[]{-1.};
	/** Action-space upper bound */
	private static final double[] uMax = new double[]{+1.};
	/** Fixed start state. If null, the start state will be drawn uniformely
	 * between the state-space bounds */
	private final double[] x0;
	
	private final double[] xtarget;
	
	private final PointMassRewardType rewardType;
	
	private final boolean randomStartState;
	
	public PointMass(final double[] x0, final double[] xtarget,
			     	 final PointMassRewardType rewardType, boolean randomStartState) {
		super(linDynSys, 1., new double[]{0.}, new double[]{1.});
		if((!ArrUtils.allGreaterOrEqual(x0, xMin)) || (!ArrUtils.allLessOrEqual(x0, xMax))) {
			throw new IllegalArgumentException("x0 must be within state-space bounds");
		}
		if((!ArrUtils.allGreaterOrEqual(xtarget, xMin)) || (!ArrUtils.allLessOrEqual(xtarget, xMax))) {
			throw new IllegalArgumentException("xtarget must be within state-space bounds");
		}
		this.x0 = x0;
		this.xtarget = xtarget;
        this.rewardType = rewardType;
        this.randomStartState = randomStartState;
	}

	public PointMass() {
		super(new LinearDynanicSystem(new double[][]{{0.}},
				new double[][]{{1.}}), 1., new double[]{0.}, new double[]{1.});
		x0 = ArrUtils.constvec(1,0.25);
		xtarget = ArrUtils.constvec(1,0.75);
        rewardType = PointMassRewardType.COST;
        randomStartState = false;
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.Environment#startState()
	 */
	@Override
	@NonNull
	public final double[] startState() {
		if(randomStartState) {
			for(int i=0; i<xDim; i++) {
				x0[i] = xMin[i] + RandUtils.nextDouble() * (xMax[i]-xMin[i]);
			}
		}
		return x0;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#reward(double[], double[], double[])
	 */
	@Override
	public final double reward(@NonNull final double[] x,
							   @NonNull final double[] u,
							   @NonNull final double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;
		
		double squaredDist = 0.;
		for(int i=0; i<xDim; i++) {
			squaredDist += Math.pow(x[i] - xtarget[i], 2);
		}
		double r = 0.;
		switch(rewardType) {
			case COST: // Negative reward (cost)
				r = -squaredDist-ArrUtils.squaredNorm(u);
				break;
			case NORMALIZED_REWARD: // Normalized reward (in [0,1])
				double xcoeff = 1.;
				double ucoeff = 1.;
	            r = ( xcoeff*Math.exp(-Math.sqrt(squaredDist))
	                  + ucoeff*Math.exp(-ArrUtils.norm(u)) )
	                / (xcoeff + ucoeff);
	            break;
			case GOAL_REWARD:
				// Goal reached ?
				if(Math.sqrt(squaredDist) < goalRadius) {
					r += 1.;
				}
				r -= ArrUtils.squaredNorm(u);
				break;
	        default:
	        	throw new IllegalArgumentException("rewardType value is incorrect !");
		}
		return r;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#isTerminal(double[], double[], double[])
	 */
	@Override
	public final boolean isTerminal(@NonNull final double[] x,
									@NonNull final double[] u,
									@NonNull final double[] xn) {
		assert x.length == xDim;
		assert u.length == uDim;
		assert xn.length == xDim;
		
		if(rewardType == PointMassRewardType.GOAL_REWARD) {
			double squaredDist = 0.;
			for(int i=0; i<xDim; i++) {
				squaredDist += Math.pow(x[i] - xtarget[i], 2);
			}
			if(Math.sqrt(squaredDist) < goalRadius) {
				return true;
			}
		}
		return false;
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getUMin()
	 */
	public double[] getUMin() {
		return uMin;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.Environment#getUMax()
	 */
	public double[] getUMax() {
		return uMax;
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.ReachingTask#getTargetState(double[])
	 */
	public double[] getTargetState(double[] x) {
		if(rewardType != PointMassRewardType.GOAL_REWARD) {
			throw new RuntimeException("rewardType must be GOAL_REWARD for the "
				+PointMass.class.getName()+" to be a "+ReachingTask.class.getName());
		}
		return xtarget.clone();
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.ReachingTask#getCurrentState(double[])
	 */
	@NonNull
	public double[] getCurrentState(@NonNull final double[] x) {
		if(rewardType != PointMassRewardType.GOAL_REWARD) {
			throw new RuntimeException("rewardType must be GOAL_REWARD for the "
				+PointMass.class.getName()+" to be a "+ReachingTask.class.getName());
		}
		return x;
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.ReachingTask#getCostFactor()
	 */
	public double getCostFactor() {
		if(rewardType != PointMassRewardType.GOAL_REWARD) {
			throw new RuntimeException("rewardType must be GOAL_REWARD for the "
				+PointMass.class.getName()+" to be a "+ReachingTask.class.getName());
		}
		return 1.;
	}
	
	/* (non-Javadoc)
	 * @see jrl.environment.ReachingTask#getGoalReward()
	 */
	public double getGoalReward() {
		if(rewardType != PointMassRewardType.GOAL_REWARD) {
			throw new RuntimeException("rewardType must be GOAL_REWARD for the "
				+PointMass.class.getName()+" to be a "+ReachingTask.class.getName());
		}
		return 1.;
	}
}
