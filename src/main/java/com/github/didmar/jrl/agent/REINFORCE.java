package com.github.didmar.jrl.agent;

import java.util.List;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.environment.Logger;
import com.github.didmar.jrl.evaluation.REINFORCEGradientEstimator;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.ParametricFunction;

/**
 * Agent that implements the Episodic REINFORCE method with optimal baseline,
 * in the discounted reward case.
 * 
 * @author Didier Marin
 */
public final class REINFORCE extends LearningAgent {

	/** Learning step */
	private final StepSize stepSize;
	/** Number of episodes to use for each policy update */
	private final int nEpiPerUpdate;
	/** Length of the policy parameters vector */
	private final int n;
	/** This computes the REINFORCE estimation of the performance gradient */
	private final REINFORCEGradientEstimator estimator;
	/** a {@link jrl_testing.environment.Logger} that stores the sample episodes */
	private final Logger log;
	
	// arrays for temporary storage to avoid mem. alloc.
	/** Used to store the product of the learning rate alpha by the gradient
	 * estimate dJ */
	private final double[] alphaMultdJ;
	
	public REINFORCE(ILogDifferentiablePolicy pol, int xDim, int uDim,
			StepSize stepSize, int nEpiPerUpdate, DiscountFactor gamma) {
		super(pol);
		this.stepSize = stepSize;
        this.nEpiPerUpdate = nEpiPerUpdate;
        // assert this.nEpiPerUpdate > 0
        n = pol.getParamsSize(); // Number of policy (and advantage approximation) parameters
        estimator = new REINFORCEGradientEstimator(pol,gamma,nEpiPerUpdate);
        log = new Logger(xDim,uDim); // Use a logger to keep the samples
        
        // init the arrays for temporary storage
        alphaMultdJ = new double[n];
	}
	
	/**
	 * Propagate the notification to the logger responsible of storing the samples.
	 * @see jrl_testing.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(@NonNull final double[] x0, int maxT) {
		log.newEpisode(x0,maxT);
	}

	/**
	 * Propagate the notification to the logger responsible of storing the samples.
	 * @see jrl_testing.environment.EnvironmentListener#receiveSample(double[], double[], double[], double, boolean)
	 */
	public final void receiveSample(@NonNull final double[] x,
									@NonNull final double[] u,
									@NonNull final double[] xn,
									double r, boolean isTerminal) {
		log.receiveSample(x,u,xn,r, false);
	}

	/**
	 * Propagate the notification to the logger responsible of storing the
	 * samples and, if the required number of sample episodes is matched,
	 * perform a policy update.
	 * @see jrl_testing.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		log.endEpisode();
        // Test if we reached the number of episode needed for an update
        if(log.getNbEpisodes() == nEpiPerUpdate) {
            // Update the step-size
            stepSize.updateStep();
            // Get the episodes from the logger
            final List<Episode> episodes = log.getEpisodes();
            // Compute the REINFORCE gradient estimation
            try {
            	@NonNull final double[] dJ
            		= estimator.computeGradientEstimation(episodes);
				// Compute the policy parameters update
	            double alpha = stepSize.getStep(); // Get the current learning rate
	            for(int i=0; i<n; i++) {
	            	// Multiply the gradient estimation by the learning rate
	            	alphaMultdJ[i] = alpha * dJ[i];
	            }
	            // Update the policy parameters
	            ((ParametricFunction) pol).updateParams( alphaMultdJ );
				// Reset the logger since we don't want to reuse these episodes
	            log.reset();
            } catch (Exception e) {
				e.printStackTrace();
				System.exit(0);
			}
        }
	}

	@Override
	@NonNull
	public final String toString() {
		return "REINFORCE";
	}
	
	

}
