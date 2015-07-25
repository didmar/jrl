package com.github.didmar.jrl.utils.plot;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;

import org.eclipse.jdt.annotation.NonNullByDefault;

/**
 * This class is used to communicate with a gnuplot instance. See
 * http://www.gnuplot.info/ for details about gnuplot. Note that gnuplot exits,
 * when the java program does. However, by sending the <tt>&quot;exit&quot;</tt>
 * command to gnuplot via {@link Gnuplot#execute(String)}, the gnuplot process
 * exits, too. In order to quit a gnuplot session, the method
 * {@link Gnuplot#close()} first sends the <tt>&quot;exit&quot;</tt> command to
 * gnuplot and then closes the communication channel.
 * <p>
 * If the default constructor does not work on your system, try to set the
 * gnuplot executable in the public static field
 * {@link com.github.didmar.jrl.utils.plot.Gnuplot#executable}.
 * 
 * This class is borrowed from the Java XCSF library.
 * 
 * @author Didier Marin
 */
@NonNullByDefault
public final class Gnuplot {

	/**
	 * Gnuplot executable on the filesystem 
	 */
	public final static String executable = getGnuplotExecutableName();

	// communication channel: console.output -> process.input
	private final PrintStream console;

	/**
	 * Default constructor executes a OS-specific command to start gnuplot and
	 * establishes the communication. If this constructor does not work on your
	 * machine, you can specify the executable in
	 * {@link com.github.didmar.jrl.utils.plot.Gnuplot#executable}.
	 * <p>
	 * <b>Windows</b><br/>
	 * Gnuplot is expected to be installed at the default location, that is
	 * 
	 * <pre>
	 * C:\&lt;localized program files directory&gt;\gnuplot\bin\pgnuplot.exe
	 * </pre>
	 * 
	 * where the <tt>Program Files</tt> directory name depends on the language
	 * set for the OS. This constructor retrieves the localized name of this
	 * directory.
	 * <p>
	 * <b>Linux</b><br/>
	 * On linux systems the <tt>gnuplot</tt> executable has to be linked in one
	 * of the default pathes in order to be available system-wide.
	 * <p>
	 * <b>Other Operating Systems</b><br/>
	 * Other operating systems are not available to the developers and comments
	 * on how defaults on these systems would look like are very welcome.
	 * 
	 * @throws IOException
	 *             if the system fails to execute gnuplot
	 */
	public Gnuplot() throws IOException {
		// start the gnuplot process and connect channels
		Process p = Runtime.getRuntime().exec(executable);
		console = new PrintStream(p.getOutputStream());
	}

	public static String getGnuplotExecutableName() {
		final String os = System.getProperty("os.name").toLowerCase();

		if (os.contains("linux")) {
			// assume that path is set
			return "gnuplot";
		}
		if (os.contains("windows")) {
			// assume default installation path, i.e.
			// <localized:program files>/gnuplot/
			String programFiles = System.getenv("ProgramFiles");
			if (programFiles == null) { // nothing found? ups.
				programFiles = "C:" + File.separatorChar + "Program Files";
			}
			// assert separator
			if (!programFiles.endsWith(File.separator)) {
				programFiles += File.separatorChar;
			}
			return programFiles + "gnuplot" + File.separatorChar
					+ "bin" + File.separatorChar + "pgnuplot.exe";
		}
		throw new RuntimeException("Operating system '" + os
					+ "' is not supported. "
					+ "If you have Gnuplot installed, "
					+ "specify the executable command via"
					+ System.getProperty("line.separator")
					+ "Gnuplot.executable "
					+ "= \"your executable\"");
	}

	/**
	 * Sends the given <code>command</code> to gnuplot. Multiple commands can be
	 * seperated with a semicolon.
	 * 
	 * @param command
	 *            the command to execute on the gnuplot process
	 */
	public final void execute(String command) {
		console.println(command);
		console.flush();
	}

	/**
	 * Exit gnuplot and close the in/out streams.
	 */
	public final void close() {
		this.execute("exit");
		console.close();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see java.lang.Object#finalize()
	 */
	@Override
	protected final void finalize() throws Throwable {
		this.close();
		super.finalize();
	}
}
