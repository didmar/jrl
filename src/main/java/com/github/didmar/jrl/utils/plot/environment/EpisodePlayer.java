package com.github.didmar.jrl.utils.plot.environment;

import java.util.ArrayList;
import java.util.List;

import com.github.didmar.jrl.utils.Episode;

/**
 * An application to display {@link Episode} given a corresponding
 * {@link EnvironmentDisplay}.
 * @author Didier Marin
 */
public final class EpisodePlayer {

	private final EnvironmentDisplay disp;
	private final List<Episode> episodes;

	/** Index of the currently played episode in the episode list */
	private int currentEpi;
	/** Time step of the currently played episode, or -1 if stopped */
	private int t;

	public EpisodePlayer(EnvironmentDisplay disp, Iterable<Episode> episodes,
			long period) {
		this.disp = disp;
		this.episodes = new ArrayList<Episode>();
		if(episodes != null) {
			for(Episode e : episodes) {
				this.episodes.add(e);
			}
		}
		currentEpi = 0;
		rewind();
	}

	public final String generateInfoText() {
		if(episodes.isEmpty()) {
			return "episode -/- step -/-";
		}
		return "episode "+(currentEpi+1)+"/"+episodes.size()
				+" step "+(t+1)+"/"+(episodes.get(currentEpi).getT()+1);
	}

	public final void previousEpisode() {
		setEpisode(currentEpi-1);
		rewind();
	}

	public final void nextEpisode() {
		setEpisode(currentEpi+1);
		rewind();
	}

	public final void setEpisode(int epi) {
		if(!episodes.isEmpty()) {
			currentEpi = epi;
			if(currentEpi < 0) {
				currentEpi = episodes.size() + currentEpi;
			} else {
				if(currentEpi >= episodes.size()) {
					currentEpi %= episodes.size();
				}
			}
			rewind();
		}
	}

	public final boolean step() {
		if(episodes.isEmpty()) {
			return true;
		}
		Episode epi = episodes.get(currentEpi);
		int T = epi.getT();
		assert t < T;
		if(t < 0) {
			disp.newEpisode(epi.getX()[0], T);
			t = 0;
		} else {
			if(t < T) {
				boolean terminated = false;
				if(t == T - 1) {
					terminated = epi.hasTerminated();
				}
				disp.receiveSample(epi.getX()[t], epi.getU()[t], epi.getXn()[t],
						epi.getR()[t], terminated);
				t++;
			} else {
				disp.endEpisode();
				return false;
			}
		}
		return true;
	}

	public final void rewind() {
		t = -1;
	}

	public final int getCurrentStep() {
		return t;
	}

	public final void setStep(int t) {
		if(t < 0) {
			throw new RuntimeException("t must be positive");
		}
		if(!episodes.isEmpty()) {
			// Hide the display before messing with the step
			disp.setVisible(false);
			Episode epi = episodes.get(currentEpi);
			int T = epi.getT();
			if(t >= T) {
				throw new RuntimeException("t must be smaller than episode duration");
			}
			// End the current episode if necessary
			if(t >= 0) {
				disp.endEpisode();
			}
			// (Re)start the current episode
			disp.newEpisode(epi.getX()[0], T);
			double[][] X = epi.getX();
			double[][] U = epi.getU();
			double[][] Xn = epi.getXn();
			double[] R = epi.getR();
			for (int k = 0; k < t; k++) {
				disp.receiveSample(X[k], U[k], Xn[k], R[k], false);
			}
			boolean terminated = false;
			if(t == T - 1) {
				terminated = epi.hasTerminated();
			}
			disp.receiveSample(X[t], U[t], Xn[t], R[t], terminated);
			// Make the display visible again
			disp.setVisible(true);
			// Set the new value of the current step
			this.t = t;
		}
	}

	public final void addEpisode(Episode e) {
		episodes.add(e);
	}

	public final void removeEpisode(int i) {
		episodes.remove(i);
	}

	public final Episode getCurrentEpisode() {
		if(episodes.isEmpty()) {
			return null;
		}
		return episodes.get(currentEpi);
	}

	public final void removeCurrentEpisode() {
		if(!episodes.isEmpty()) {
			episodes.remove(currentEpi);
			if(currentEpi > 0 && currentEpi == episodes.size()) {
				currentEpi--;
			}
			rewind();
		}
	}
}
