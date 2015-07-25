package com.github.didmar.jrl.utils.plot;

import java.io.IOException;

import com.github.didmar.jrl.utils.Console;

/**
 * A Python console ready to plot using Matplotlib library.
 * @author Didier Marin
 */
public final class Matplot extends Console {

	public Matplot() throws IOException {
		super("ipython");
		setExitCommand("exit");
		execute("from numpy import *");
		execute("from matplotlib import *");
		execute("from matplotlib.pyplot import *");
		execute("ion()");
	}

}
