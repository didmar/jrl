/**
 * Policies are probability distributions over actions given a state, which can
 * used by agents ({@link com.github.didmar.jrl.agent.PolicyAgent}) to choose
 * their actions.
 * Parametric policies ({@link jrl_testing.policy.ParametricPolicy}) can be optimized
 * by their agent : i.e. for policy gradient methods such as REINFORCE (Agent 
 * {@link com.github.didmar.jrl.agent.REINFORCE}) will need a policy which implements 
 * {@link jrl_testing.policy.ILogDifferentiablePolicy}.
 */
@org.eclipse.jdt.annotation.NonNullByDefault
package com.github.didmar.jrl.policy;