package com.github.didmar.jrl.environment;

public interface ReachingTask {
	public double[] getTargetState(double[] x);
	public double[] getCurrentState(double[] x);
	public double getCostFactor();
	public double getGoalReward();
}
