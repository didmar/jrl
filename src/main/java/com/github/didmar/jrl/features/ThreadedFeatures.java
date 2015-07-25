package com.github.didmar.jrl.features;

import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.eclipse.jdt.annotation.NonNull;
import org.eclipse.jdt.annotation.Nullable;

// TODO extract the thread pool code

/**
 * Aggregate multiple features that are computed in parallel
 * 
 * @author Didier Marin
 */
public class ThreadedFeatures extends Features {

	/** maximum idle time of the thread pool (25 sec) */
	private final static long MAX_IDLE_TIME = 25000;
	
	private final static long TIMERTASK_PERIOD = 5000;
	/** Array of Features to be computed in parallel */
	private final Features[] feats;
	
	private final int nbFeats;
	/** Share the input between jobs */
	@Nullable double[] sharedX = null;
	/** nbFeats-1 workers */
	private final FeaturesJob[] jobs;
	/** enables synchronization */
	protected final CyclicBarrier synchronization;
	/** a fixed size thread pool */
	protected ExecutorService pool;
	/** last time the threadpool was used */
	protected long updateTimestamp = System.currentTimeMillis();

	/**
	 * @param inDim
	 * @param outDim
	 */
	@SuppressWarnings("null")
	public ThreadedFeatures(final Features[] feats) {
		super(feats[0].inDim, computeOutDim(feats));
		this.feats = feats;
		nbFeats =  feats.length;
		// initialize worker jobs: n processors, 1=main, 2..n=worker(s)
		final ExecutorService _pool = Executors.newFixedThreadPool(nbFeats - 1);
		if(_pool == null) throw new RuntimeException("Could not create thread pool");
		pool = _pool;
		new Timer().scheduleAtFixedRate(new ShutdownTask(),
				TIMERTASK_PERIOD, TIMERTASK_PERIOD);
		synchronization = new CyclicBarrier(nbFeats);
		// n clusters for n threads
		jobs = new FeaturesJob[nbFeats - 1];
		for (int i = 1; i < nbFeats; i++) {
			jobs[i - 1] = new FeaturesJob(feats[i]);
		}
	}

	private static int computeOutDim(Features[] feats) {
		int cpt = 0;
		for(Features f : feats) {
			cpt += f.outDim;
		}
		return cpt;
	}

	/* (non-Javadoc)
	 * @see jrl.features.Features#phi(double[], double[])
	 */
	@Override
	public void phi(@NonNull double[] x, @NonNull double[] y)
			throws IllegalArgumentException {
		assert x != null;
		assert y != null;
		
		if(x.length != inDim ){
			throw new IllegalArgumentException("x must have length inDim");
		}
		if(y.length != outDim ){
			throw new IllegalArgumentException("y must have length outDim");
		}
		
		this.sharedX = x;
		
		if (pool.isShutdown()) {
			this.reviveThreadpool();
		}
		updateTimestamp = System.currentTimeMillis();

		try {
			// submit n-1 jobs to workers
			for(final FeaturesJob job : jobs) {
				pool.execute(job);
			}

			// farmer does computations, too
			feats[0].phi(x, y);
			
			// synchronize with workers to assure everything is done
			synchronization.await();
			// now, farmer and all workers are done
			
			// copy the results
			int cpt = feats[0].outDim;
			for(final FeaturesJob job : jobs) {
				int n = job.feat.outDim;
				System.arraycopy(job.y, 0, y, cpt, n);
				cpt += n;
			}

		} catch (final InterruptedException e) {
			e.printStackTrace();
		} catch (final BrokenBarrierException e) {
			e.printStackTrace();
		}
	}

	private class FeaturesJob implements Runnable {

		final Features feat;
		public final double[] y;

		public FeaturesJob(final Features feat) {
			this.feat = feat;
			y = new double[feat.outDim];
		}

		/*
		 * (non-Javadoc)
		 * @see java.lang.Runnable#run()
		 */
		public void run() {
			// FIXME avoid using a RuntimeException
			if(sharedX != null) {
				@NonNull final double[] x = sharedX;
				// compute the features
				feat.phi(x, y);
				// synchronize with main thread
				try {
					synchronization.await();
				} catch (final InterruptedException e) {
					e.printStackTrace();
				} catch (final BrokenBarrierException e) {
					e.printStackTrace();
				}
			} else {
				throw new RuntimeException("sharedX is null");
			}
		}
	}
	
	class ShutdownTask extends TimerTask {

		/*
		 * (non-Javadoc)
		 * 
		 * @see java.util.TimerTask#run()
		 */
		@Override
		public void run() {
			if (pool.isShutdown()) {
				this.cancel();
				return;
			}
			long idletime = System.currentTimeMillis() - updateTimestamp;
			if (idletime > MAX_IDLE_TIME) {
				// pool is idle, shutdown
				ThreadedFeatures.this.shutdownThreadpool();
				this.cancel();
			} // else { pool is still in use }
		}

	}
	
	public void shutdownThreadpool() {
		pool.shutdownNow();
	}
	
	/**
	 * Revives a dead threadpool, if this matchset is constructed with
	 * multithreading enabled.
	 * @throws Exception if the thread pool creation fails
	 */
	void reviveThreadpool() {
		// FIXME redundant code
		final ExecutorService _pool = Executors.newFixedThreadPool(nbFeats - 1);
		if(_pool == null) throw new RuntimeException("Could not create thread pool");
		pool = _pool;
		new Timer().scheduleAtFixedRate(new ShutdownTask(),
					TIMERTASK_PERIOD, TIMERTASK_PERIOD);
	}
	
	/* (non-Javadoc)
	 * @see jrl.features.Features#isNormalized()
	 */
	@Override
	public boolean isNormalized() {
		// TODO a determiner
		return false;
	}

}
