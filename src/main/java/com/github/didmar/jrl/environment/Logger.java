package com.github.didmar.jrl.environment;

import java.util.ArrayList;
import java.util.List;

import org.eclipse.jdt.annotation.NonNull;
import org.eclipse.jdt.annotation.Nullable;

import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * Listen to the environment and store the (x,u,xn,r) samples.
 *
 * @author Didier Marin
 */
public final class Logger implements EnvironmentListener {

	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;
	/** List of finished episodes. The current episode is finished once
	 * {@link #endEpisode()} is called **/
	private final List<Episode> episodes;
	/** The current episode */
	@Nullable private Episode currentEpi = null;
	/** Verbosity */
	private final boolean verbose;

	public Logger(int xDim, int uDim) {
		this.xDim = xDim;
		this.uDim = uDim;
		episodes = new ArrayList<Episode>();
		verbose = false;
	}

	public Logger(int xDim, int uDim, int nbEpisodes) {
		this.xDim = xDim;
		this.uDim = uDim;
		episodes = new ArrayList<Episode>(nbEpisodes);
		this.verbose = false;
	}

	public Logger(int xDim, int uDim, boolean verbose) {
		this.xDim = xDim;
		this.uDim = uDim;
		episodes = new ArrayList<Episode>();
		this.verbose = verbose;
	}

	public Logger(int xDim, int uDim, int nbEpisodes, boolean verbose) {
		this.xDim = xDim;
		this.uDim = uDim;
		episodes = new ArrayList<Episode>(nbEpisodes);
		this.verbose = verbose;
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#newEpisode(double[], int)
	 */
	public final void newEpisode(@NonNull double[] x0, int maxT) {
		if(currentEpi != null) {
			throw new RuntimeException("newEpisode called before endEpisode");
		}
		currentEpi = new Episode(maxT,xDim,uDim);
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#receiveSample(double[], double[], double[], double)
	 */
	public final void receiveSample(@NonNull double[] x,
									@NonNull double[] u,
									@NonNull double[] xn,
									double r,
									boolean isTerminal) {
		if(verbose) {
			System.out.print("x="+ArrUtils.toString(x)+" u="+ArrUtils.toString(u)
					+" xn="+ArrUtils.toString(xn)+" r="+r);
			if(isTerminal) {
				System.out.println(" (terminal)");
			} else {
				System.out.println();
			}
		}
		if(currentEpi != null) {
			currentEpi.addSample(x,u,xn,r,isTerminal);
		} else {
			throw new RuntimeException("receiveSample called before newEpisode");
		}
	}

	/* (non-Javadoc)
	 * @see jrl.environment.EnvironmentListener#endEpisode()
	 */
	public final void endEpisode() {
		if(currentEpi != null) {
			// resize the samples array to the current duration
			currentEpi.setDuration();
	        episodes.add(currentEpi);
	        currentEpi = null;
		} else {
			throw new RuntimeException("endEpisode called before newEpisode");
		}
	}

	/**
	 * Clears all the logged episodes, including the current episode.
	 */
	public final void reset() {
        episodes.clear();
        currentEpi = null;
	}

	/**
	 * @return the number of (terminated) logged episodes
	 */
	public final int getNbEpisodes() {
		return episodes.size();
	}

	/**
	 * @return the (terminated) logged episodes
	 */
	public final List<Episode> getEpisodes() {
		return episodes;
	}

	/**
	 * @return an array that contains the duration of each (terminated) logged episode
	 */
	public final int[] episodesDuration() {
        int[] T = new int[episodes.size()];
        for(int i=0; i<T.length; i++) {
            T[i] = episodes.get(i).getT();
        }
        return T;
	}

	/**
	 * Compute the average reward of the (terminated) logged episodes.
	 * @return an array that contains the average reward of each logged episode
	 */
	public final double[] averageReward() {
        double[] R = new double[episodes.size()];
        for(int i=0; i<R.length; i++) {
        	R[i] = episodes.get(i).averageReward();
        }
        return R;
	}

	/**
	 * Compute the discounted reward of the (terminated) logged episodes.
	 * @param gamma  discount factor
	 * @return       an array that contains the discounted reward of each logged episode
	 */
	public final double[] discountedReward(DiscountFactor gamma) {
		double[] R = new double[episodes.size()];
        for(int i=0; i<R.length; i++) {
        	R[i] = episodes.get(i).discountedReward(gamma);
        }
        return R;
	}

	@Override
	@NonNull
	public final String toString() {
		final String s = episodes.toString();
		if(s==null) return "";
		return s;
	}
}
