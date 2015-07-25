package com.github.didmar.jrl.agent.ac;

import org.eclipse.jdt.annotation.NonNull;

import com.github.didmar.jrl.agent.LearningAgent;
import com.github.didmar.jrl.evaluation.valuefunction.LinearQFunction;
import com.github.didmar.jrl.evaluation.valuefunction.LinearVFunction;
import com.github.didmar.jrl.evaluation.vflearner.lstd.ILSTDAV;
import com.github.didmar.jrl.features.CompatibleFeatures;
import com.github.didmar.jrl.features.Features;
import com.github.didmar.jrl.policy.ILogDifferentiablePolicy;
import com.github.didmar.jrl.stepsize.StepSize;
import com.github.didmar.jrl.utils.DiscountFactor;

// TODO add another convergence criterion : when the norm of the update of
// advantage and state value parameters is below a threshold epsilon
/**
 * Natural Actor-Critic. See Peters and Schaal 2008 "Natural actor-critic".
 * 
 * @author Didier Marin
 */
public final class NAC extends LearningAgent {

	private final ILSTDAV ilstdav;
	/** Linear advantage function approximator for the Critic part */
	private final LinearQFunction aFunction;
	/** Linear state value function approximator for the Critic part */
	private final LinearVFunction vFunction;
	/** Learning step */
	private final StepSize stepSize;
	/** Forget factor */
	private final DiscountFactor kappa;
	/** Number of (x,u,r,xn) samples to use for each update */
	private final int nbSamplesBeforeUpdate;
	/** Number of (x,u,r,xn) samples collected since the last update */
	private int nbSampleSinceLastUpd;
	
	public NAC(final ILogDifferentiablePolicy pol,
			   final Features stateFeat,
			   final StepSize stepSize,
			   final DiscountFactor gamma,
			   final DiscountFactor lambda,
			   final DiscountFactor kappa,
			   int nbSamplesBeforeUpdate,
			   double diagAinv0,
			   int xDim,
			   int uDim) throws Exception {
		super(pol);
		aFunction = new LinearQFunction(new CompatibleFeatures(pol, xDim, uDim),
				xDim, uDim);
		vFunction = new LinearVFunction(stateFeat);
		ilstdav = new ILSTDAV(aFunction, vFunction, gamma, lambda,
				nbSamplesBeforeUpdate, diagAinv0);
		this.stepSize = stepSize;
		this.kappa = kappa;
		this.nbSamplesBeforeUpdate = nbSamplesBeforeUpdate;
		nbSampleSinceLastUpd = 0;
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		// Nothing to do
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(@NonNull final double[] x0, int maxT) {
		// Nothing to do
	}

	/* (non-Javadoc)
	 * @see jrl_testing.environment.EnvironmentListener#receiveSample(double[], double[], double[], double, boolean)
	 */
	public final void receiveSample(@NonNull final double[] x,
									@NonNull final double[] u,
									@NonNull final double[] xn,
									double r, boolean isTerminal) {
		// Update the step-size
        stepSize.updateStep();
        // Update the sample counter
        nbSampleSinceLastUpd++;
        
        // *** Critic part
        ilstdav.receiveSample(x, u, xn, r, isTerminal);
        
        // *** Actor part
        // Check if the gradient w has converged
        boolean converged = false;
        if(nbSamplesBeforeUpdate != 0) {
            if(nbSampleSinceLastUpd >= nbSamplesBeforeUpdate) {
                converged = true;
            }
        }
//        if(epsilon > 0.) {
//            delta_w = linalg.norm(self.w - self.w_old)
//            delta_v = linalg.norm(self.v - self.v_old)
//            print "delta_w=", delta_w, "delta_v=", delta_v
//            if delta_w < self.epsilon:
//                converged = True
//                self.w_old = self.w
//                self.v_old = self.v
//        }
        
        // If it has converged
        if(converged) {
            // Perform a policy update
        	double alpha = stepSize.getStep();
        	double[] w = aFunction.getParams().clone();
        	for(int i=0; i<w.length; i++) {
        		w[i] *= alpha;
        	}
        	((ILogDifferentiablePolicy)pol).updateParams(w);
            // Partially forget statistics
            ilstdav.applyForgetFactor(kappa);
            // Reset the sample counter
            nbSampleSinceLastUpd = 0;
        }
	}

	public final LinearQFunction getAFunction() {
		return aFunction;
	}
	
	public final LinearVFunction getVFunction() {
		return vFunction;
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	@NonNull
	public final String toString() {
		return "Natural Actor-Critic";
	}
}
