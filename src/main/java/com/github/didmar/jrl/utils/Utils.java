package com.github.didmar.jrl.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.Callable;

import org.eclipse.jdt.annotation.NonNull;
import org.eclipse.jdt.annotation.Nullable;

/**
 * Various useful static methods, some of which are borrowed from the JavaXCSF library.
 * In many functions no safety checks are done to improve performance,
 * unless assertions are enabled (run java with the <tt>-enableassertions</tt>
 * or <tt>-ea</tt> switch).
 * 
 * @author Didier Marin
 */
public final class Utils {
	
	/**
	 * The difference between 1 and the smallest exactly representable number
	 * greater than one. Gives an upper bound on the relative error due to
	 * rounding of floating point numbers.
	 */
	public final static double MACHEPS = getMacheps(); // 2E-16

	/**
	 * Get MACHEPS for the executing machine.
	 */
	public final static double getMacheps() {
		double macheps = 1;
		do {
			macheps /= 2;
		} while (1 + macheps / 2 != 1);
		return macheps;
	 }
	
	private final static double heavyside_KAPPA=500.;

	public final static void heavyside(double[] src, double[] dest, int n) {
		assert n > 0;
		assert src != null;
		assert src.length == n;
		assert dest != null;
		assert dest.length == n;
		
	    for(int i=0; i<n; i++) {
	    	double tmp = heavyside_KAPPA*src[i];
	    	if(tmp > 200) {
	    		dest[i] = src[i];
	    	} else {
	    		dest[i] = Math.log(Math.exp(tmp) + 1.) / heavyside_KAPPA;
	    	}
		}
	}	
	
	/**
	 * Sort an array and store the indices of the sorted elements. 
	 * Preconditions : index must be allocated, with the same length as array.
	 * @param array the array to be sorted
	 * @param index indices of the sorted elements in the original array
	 */
	public final static void quicksort(final double[] array,
			                           final int[] index) {
		assert array != null;
		assert index != null;
		assert array.length == index.length;
		
		for(int i=0; i<index.length; i++) {
			index[i] = i;
		}
	    quicksort(array, index, 0, index.length-1);
	}

	// quicksort a[left] to a[right]
	private final static void quicksort(final double[] a, final int[] index,
								  		int left, int right){
		assert a != null;
		assert index != null;
		
	    if(right <= left) return;
	    int i = partition(a, index, left, right);
	    quicksort(a, index, left, i-1);
	    quicksort(a, index, i+1, right);
	}

	// partition a[left] to a[right], assumes left < right
	private final static int partition(final double[] a,
								 	   final int[] index,
								 	   int left, int right) {
		assert a != null;
		assert index != null;
		assert left < right;
		
	    int i = left - 1;
	    int j = right;
	    while (true) {
	        while (less(a[++i], a[right]))      // find item on left to swap
	        	continue;                       // a[right] acts as sentinel
	        while (less(a[right], a[--j]))      // find item on right to swap
	            if (j == left) break;           // don't go out-of-bounds
	        if (i >= j) break;                  // check if pointers cross
	        exch(a, index, i, j);               // swap two elements into place
	    }
	    exch(a, index, i, right);               // swap with partition element
	    return i;
	}

	// is x < y ?
	private final static boolean less(double x, double y) {
	    return x < y;
	}
	
	// exchange a[i] and a[j]
	private final static void exch(double[] a, int[] index, int i, int j) {
		assert a != null;
		assert index != null;
		
		double swap = a[i];
	    a[i] = a[j];
	    a[j] = swap;
	    int b = index[i];
	    index[i] = index[j];
	    index[j] = b;
	}
	
	/**
	 * Returns <code>true</code> if two values are equal within a tolerance.
	 * 
	 * @param a
	 *            the first value
	 * @param b
	 *            the second value
	 * @param tol
	 * 			  tolerance parameter
	 * @return <code>true</code>, if <code>a</code> and <code>b</code> contain
	 * 			the same values within tolerance <code>tol</code>;
	 * 			<code>false</code> otherwise.
	 */
	public final static boolean allClose(double a, double b, double tol) {
		if (Math.abs(a - b) > tol) {
			return false;
		}
		return true;
	}
	
	public final static double boundAngle(double alpha) {
		return alpha % (2*Math.PI);
	}
	
	public final static void boundAngles(final double[] alphas) {
		assert alphas != null;
		
		for (int i = 0; i < alphas.length; i++) {
			alphas[i] = boundAngle(alphas[i]);
		}
	}
	
	/**
	 * Prompt the user for an item within a list of choices 
	 * @param choices list of choices 
	 * @return index of the chosen item 
	 * @throws Exception if an IO error occurs
	 */
	public final static int chooseOne(final String[] choices) throws Exception {
		// print possible choices
		int i = 0;
		for(String choice : choices) {
			System.out.println("["+i+"] "+choice);
			i++;
		}
		//  open up standard input
		@NonNull final BufferedReader br =
				new BufferedReader(new InputStreamReader(System.in));
		// FIXME find a cleaner way to use untilNotNull !
	    @NonNull Integer choice = untilNotNull(
	    		new Callable<Integer>() {public @Nullable Integer call() throws Exception {
	    			return getIntegerFromUser(br);}}
	    		);
		return choice.intValue();
	}
	
	final static @Nullable Integer getIntegerFromUser(final BufferedReader br)
			throws IOException {
		System.out.print("> ");
		@Nullable final String s = br.readLine();
		try {
			return Integer.parseInt(s); // Valid choice
		} catch (Exception e) {
			System.err.println("Not a valid choice, try again");			
		}
		return null; // Invalid choice
	}
	
	public static final <T> T untilNotNull(final Callable<T> function)
			throws Exception {
		@Nullable T x = function.call();
		while(x==null) {
			x = function.call();
		}
		return x;
	}

	public static void waitForKeypress() {
		System.out.println("Press a key to terminate...");
		try {
			System.in.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
