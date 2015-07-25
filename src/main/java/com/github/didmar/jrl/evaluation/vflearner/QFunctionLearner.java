package com.github.didmar.jrl.evaluation.vflearner;

import com.github.didmar.jrl.evaluation.valuefunction.QFunction;

/**
 * Interface for all state-action value function or advantage function learning
 * methods.
 * @author Didier Marin
 */
public interface QFunctionLearner {
	public QFunction getQFunction();
}
