/**
 * Environments are Markovian Decision Processes that agents can interact with.
 * These interactions can be observed by some listeners through the
 * {@link com.github.didmar.jrl.environment.EnvironmentListener} interface. In particular, the
 * learning agents should use this interface to get the (x,u,xn,r) samples
 * produced by their interactions.
 * @see com.github.didmar.jrl.agent.Agent
 */
@org.eclipse.jdt.annotation.NonNullByDefault
package com.github.didmar.jrl.environment;