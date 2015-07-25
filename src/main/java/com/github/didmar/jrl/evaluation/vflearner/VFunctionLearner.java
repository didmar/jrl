package com.github.didmar.jrl.evaluation.vflearner;

import com.github.didmar.jrl.evaluation.valuefunction.VFunction;

/**
 * Interface for all state value function learning methods.
 * @author Didier Marin
 */
public interface VFunctionLearner {
	public VFunction getVFunction();
}
