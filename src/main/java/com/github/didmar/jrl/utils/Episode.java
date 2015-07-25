package com.github.didmar.jrl.utils;

import java.io.*;
import java.lang.Math;

import org.eclipse.jdt.annotation.NonNull;
import org.eclipse.jdt.annotation.Nullable;

import com.github.didmar.jrl.utils.array.ArrUtils;

// TODO add methods to read/write in XML (using XMLEncoder and XMLDecoder)
/**
 * An episode is a list of (x,u,xn,r) samples.
 * @author Didier Marin
 */
public final class Episode implements Serializable {

	private static final long serialVersionUID = 1615636284757104553L;
	
	/** Maximum length of the episode */
	private final int maxT;
	/** State-space dimension */
	private final int xDim;
	/** Action-space dimension */
	private final int uDim;
	/** States */
	private double[][] xs;
	/** Actions */
	private double[][] us;
	/** Next states */
	private double[][] xns;
	/** Rewards */
	private double[]   rs;
	/** Indicates if the episode ended on a terminal tuple or not */
	private boolean terminated;
	/** Current length of the episode */
    int T;
	
    public Episode(int maxT, int xDim, int uDim) {
    	this.maxT = maxT;
    	this.xDim = xDim;
        this.uDim = uDim;
        // Allocate the sample arrays to the maximum length
    	xs = new double[maxT][xDim];
        us = new double[maxT][uDim];
        xns = new double[maxT][xDim];
        rs = new double[maxT];
        terminated = false;
        T = 0;
    }
    
    /**
     * Add a (x,u,xn,r) samples add to the list.
     * @throws Exception if the maximum episode length is reached 
     * 
     */
    public final void addSample(double[] x, double[] u, double[] xn, double r,
    							boolean isTerminal) {
        if(T >= maxT) {
        	throw new IllegalArgumentException("Cannot add sample : max duration reached");
        }
        if(isTerminal && terminated) {
        	throw new RuntimeException("Episode already terminated");
        }
        System.arraycopy(x,  0, this.xs[T],  0, xDim);
    	System.arraycopy(u,  0, this.us[T],  0, uDim);
    	System.arraycopy(xn, 0, this.xns[T], 0, xDim);
        this.rs[T] = r;
        terminated = isTerminal;
        T++;
    }
    
    // TODO find a better name for that function
    /**
     * Resize the samples arrays x, u, xn and r to match the episode length T 
     */
    public final void setDuration() {
        xs = (double[][]) ArrUtils.resizeArray(xs,T);
        us = (double[][]) ArrUtils.resizeArray(us,T);
        xns = (double[][]) ArrUtils.resizeArray(xns,T);
        rs = (double[]) ArrUtils.resizeArray(rs,T);
    }
    
    /**
     * Compute the average reward of the episode 
     * @return average reward of the episode
     */
    public final double averageReward() {
        return ArrUtils.mean(rs);
    }
    
    /**
     * Compute the discounted reward of the episode 
     * @return discounted reward of the episode
     */
    public final double discountedReward(DiscountFactor gamma) {
    	double R = 0;
        for(int i=0; i<T; i++) {
        	R += Math.pow(gamma.value,i)*rs[i];
        }
    	return R;
    }
    
    public final double[][] getX() {
		return xs;
	}

	public final double[][] getU() {
		return us;
	}

	public final double[] getR() {
		return rs;
	}

	public final double[][] getXn() {
		return xns;
	}

	public final int getT() {
		return T;
	}
	
	public final boolean hasTerminated() {
		return terminated;
	}

	public final void writeToTextFile(String filename, String description)
			throws IOException {
		@NonNull final File file = new File(filename);
		@NonNull final FileWriter fstream = new FileWriter(file);
		@NonNull final BufferedWriter out = new BufferedWriter(fstream);
        // Write a header
        // FIXME si la description fait plus d'une ligne, mettre des
        // '#' à chaque ligne !!! 
        out.write("# "+description+"\n");
        out.write("# xDim="+xDim+"\n# uDim="+uDim+"\n");
        // The episode data
        for(int i=0; i<T; i++) {
        	for(int j=0; j<xs[i].length; j++) {
				out.write(Double.toString(xs[i][j])+" ");
        	}
        	for(int j=0; j<us[i].length; j++) {
				out.write(Double.toString(us[i][j])+" ");
        	}
        	for(int j=0; j<xns[i].length; j++) {
        		out.write(Double.toString(xns[i][j])+" ");
        	}
        	out.write(Double.toString(rs[i]));
        	if(i < T-1) {
        		out.write("\n");
        	}
        }
        out.close();
        fstream.close();
	}
	
	public final void writeToBinaryFile(File file) throws Exception {
		// Create file output stream.
        FileOutputStream fstream = new FileOutputStream(file);
        try {
           // Create object output stream.
           ObjectOutputStream ostream = new ObjectOutputStream(fstream);
           try {
              // Write object.
              ostream.writeObject(this);
              ostream.flush();
           } finally {
              // Close object stream.
              ostream.close();
           }
        } finally {
           // Close file stream.
           fstream.close();
        }
	}
	
	public final void writeToBinaryFile(String filename) throws Exception {
		writeToBinaryFile(new File(filename));
	}
	
	public static final Episode readFromBinaryFile(File file) throws Exception {
		// Create file input stream
        @NonNull final FileInputStream fstream = new FileInputStream(file);
        // Create object output stream
        @NonNull final ObjectInputStream istream = new ObjectInputStream(fstream);
        // Read object
        @Nullable final Episode e = (Episode) istream.readObject();
	    // Close object stream
	    istream.close();
	    // Close file stream
        fstream.close();
        if(e == null) {
        	throw new Exception("Could not read episode");
        }
        return e;
	}
	
	public static final Episode readFromBinaryFile(String filename)
			throws Exception {
		return readFromBinaryFile(new File(filename));
	}
	
	@SuppressWarnings("null")
	@Override
	@NonNull
	public final String toString() {
		String s = "";
		for(int i=0; i<T; i++) {
			s += "x="+ArrUtils.toString(xs[i])
			     + " u="+ArrUtils.toString(us[i])
			     + " xn="+ArrUtils.toString(xns[i])
			     + " r="+rs[i]
			     + "\n";
		}
		return s;
	}
	
	
}
