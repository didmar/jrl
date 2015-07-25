package com.github.didmar.jrl.policy;

import com.github.didmar.jrl.evaluation.valuefunction.QFunction;
import com.github.didmar.jrl.policy.Policy;

/**
 * An interface for policies based on a QFunction, such as
 * {@link BoltzmannPolicyOverQ}. Useful for the code of learning agent such as
 * {@link SARSA}.
 * @author Didier Marin
 */
public interface QFunctionBasedPolicy extends Policy {

	QFunction getQFunction();

}
