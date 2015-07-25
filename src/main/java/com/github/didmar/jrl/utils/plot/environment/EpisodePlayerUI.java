package com.github.didmar.jrl.utils.plot.environment;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.awt.event.KeyEvent;
import java.util.Arrays;
import java.util.Timer;
import java.util.TimerTask;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenuBar;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JSlider;
import javax.swing.JToolBar;
import javax.swing.KeyStroke;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.plot.EpisodeStatsUI;

// TODO FPS fixe pour zapper certaines update quand on va vite
// TODO boite déroulante pour choisir rapidement l'épisode
// TODO quand on change d'épisode, on retourne pas au début !
// TODO possibilité d'ouvrir un log dans une nouvelle fenetre

@SuppressWarnings("serial")
/**
 * A user interface to replay episodes through a {@link EpisodePlayer}.
 * @author Didier Marin
 */
public final class EpisodePlayerUI extends JFrame implements ActionListener,
	ItemListener, ChangeListener {

	private enum PlayerState {
		STOPPED, PAUSED, PLAYING;
	}

	private final JFileChooser fileChooser = new JFileChooser();
	private final JMenuBar menuBar;
	private final JMenu fileMenu;
	private final JMenuItem openItem;
	private final JMenuItem openInNewWindowItem;
	private final JMenuItem saveItem;
	private final JMenuItem quitItem;
	private final JMenuItem closeItem;
	private final JMenu viewMenu;
	private final JMenuItem statsItem;
	private final JToolBar toolBar;
	private final JToolBar episodeBar;
	private final JButton playButton = new JButton("Play/Pause");
	private final JButton stepButton = new JButton("Step");
	private final JButton stopButton = new JButton("Stop");
	private final JButton prevButton = new JButton("Prev");
	private final JButton nextButton = new JButton("Next");
	private final JCheckBox loopBox = new JCheckBox("Loop",true);
	private final JCheckBox autoplayBox = new JCheckBox("Autoplay",true);
	private final JSlider speedSlider = new JSlider(JSlider.HORIZONTAL, 1, 9, 5);
	private final JSlider stepSlider = new JSlider(JSlider.HORIZONTAL, 0);
	private final JLabel  infoLabel  = new JLabel();

	private final String title;

	private final EpisodePlayer player;
	private final PlayerTimer playerTimer;
	/** State of the player (playing, paused or stopped) */
	private PlayerState state = PlayerState.STOPPED;

	private final EnvironmentDisplay disp;

	private final Iterable<Episode> episodes;

	private boolean loop;

	private final long basePeriod;

	private final DiscountFactor gamma;

	public EpisodePlayerUI(String title, EnvironmentDisplay disp, Iterable<Episode> episodes,
			long period, DiscountFactor gamma) {
		super(title);
		if(period < 0) {
			throw new IllegalArgumentException("period must be greater or equal to 0");
		}
		this.title = title;
		this.disp = disp;
		this.episodes = episodes;

		loop = loopBox.isSelected();
		basePeriod = period;

		this.gamma = gamma;

		// Initialize the episode player
		player = new EpisodePlayer(disp, episodes, period);
		step();

		// Listener stuff
		playButton.addActionListener(this);
		stepButton.addActionListener(this);
		stopButton.addActionListener(this);
		prevButton.addActionListener(this);
		nextButton.addActionListener(this);
		loopBox.addItemListener(this);
		autoplayBox.addItemListener(this);
		speedSlider.addChangeListener(this);
		stepSlider.addChangeListener(this);

		// Create the menu bar.
		menuBar = new JMenuBar();

		// Build the File menu.
		fileMenu = new JMenu("File");
		fileMenu.setMnemonic(KeyEvent.VK_F);
		fileMenu.getAccessibleContext().setAccessibleDescription("File menu");
		menuBar.add(fileMenu);
		// Add Open item
		openItem = new JMenuItem(new OpenAction(player));
		openItem.setAccelerator(KeyStroke.getKeyStroke(
		        KeyEvent.VK_O, ActionEvent.CTRL_MASK));
		openItem.getAccessibleContext().setAccessibleDescription(
		        "Open an Episode data file");
		fileMenu.add(openItem);
		// Add Open in new window item
		openInNewWindowItem = new JMenuItem(new OpenInNewWindowAction());
		openInNewWindowItem.setAccelerator(KeyStroke.getKeyStroke(
		        KeyEvent.VK_T, ActionEvent.CTRL_MASK));
		openInNewWindowItem.getAccessibleContext().setAccessibleDescription(
		        "Open an Episode data file in a new window");
		fileMenu.add(openInNewWindowItem);
		// Add Save as item
		saveItem = new JMenuItem(new SaveAsAction(player));
		saveItem.setAccelerator(KeyStroke.getKeyStroke(
		        KeyEvent.VK_S, ActionEvent.CTRL_MASK));
		fileMenu.add(saveItem);
		// Add Close item
		closeItem = new JMenuItem(new CloseAction(player));
		closeItem.setAccelerator(KeyStroke.getKeyStroke(
		        KeyEvent.VK_W, ActionEvent.CTRL_MASK));
		fileMenu.add(closeItem);
		// Add Quit item
		quitItem = new JMenuItem(new QuitAction());
		quitItem.setAccelerator(KeyStroke.getKeyStroke(
		        KeyEvent.VK_Q, ActionEvent.CTRL_MASK));
		fileMenu.add(quitItem);

		// Build the View menu.
		viewMenu = new JMenu("View");
		viewMenu.setMnemonic(KeyEvent.VK_V);
		viewMenu.getAccessibleContext().setAccessibleDescription("View menu");
		menuBar.add(viewMenu);
		// Add stats item
		statsItem = new JMenuItem(new ShowStatsAction());
		statsItem.setAccelerator(KeyStroke.getKeyStroke(
		        KeyEvent.VK_T, ActionEvent.CTRL_MASK));
		statsItem.getAccessibleContext().setAccessibleDescription(
		        "Show the episodes stats");
		viewMenu.add(statsItem);

		// Set the menu bar
		setJMenuBar(menuBar);

		// Change the layout to BorderLayout
		setLayout(new BorderLayout());

		// Create the toolbar
        toolBar = new JToolBar();
        toolBar.add(playButton);
        toolBar.add(stepButton);
        toolBar.add(stopButton);
        toolBar.add(prevButton);
        toolBar.add(nextButton);
        toolBar.add(loopBox);
        toolBar.add(autoplayBox);
        toolBar.add(speedSlider);
        speedSlider.setPaintTicks(true);
        speedSlider.setMajorTickSpacing(1);
        speedSlider.setSnapToTicks(true);

        // Create the episode bar
        episodeBar = new JToolBar();
        episodeBar.add(infoLabel);
        episodeBar.add(stepSlider);
        stepSlider.setMinimum(0);
        if(player.getCurrentEpisode() != null) {
        	stepSlider.setMaximum(player.getCurrentEpisode().getT());
        } else {
        	stepSlider.setMaximum(1);
        }
        stepSlider.setPaintTicks(true);
        stepSlider.setMajorTickSpacing(1);
        stepSlider.setSnapToTicks(true);

		// Add the components to the content pane
		Container pane = getContentPane();
		pane.add(toolBar, BorderLayout.PAGE_START);
		pane.add(disp, BorderLayout.CENTER);
		pane.add(episodeBar, BorderLayout.SOUTH);

		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setSize(640,480);

		// Initialize the player timer
		playerTimer = new PlayerTimer(this, period);
		// If autoplay, start playing
		if(autoplayBox.isSelected()) {
			play();
		} else {
			stop();
		}

		// Once everything is set up, make the GUI visible
		setVisible(true);
	}

	private final void play() {
		setState( PlayerState.PLAYING );
	}

	public final void stop() {
		stepSlider.setEnabled(false);
		player.rewind();
		step();
		setState( PlayerState.STOPPED );
		//stepSlider.setValue(0);
		stepSlider.setEnabled(true);
	}

	public final void pause() {
		setState( PlayerState.PAUSED );
	}

	private final void setState(PlayerState state) {
		System.out.println("state="+state.toString());
		this.state = state;
	}

	public final void step() {
		// Play one step
		if(!player.step()) {
			stop();
			if(loop) {
				play();
			}
		}
		// Update status information
		updateStatus();
	}

	private final void updateStatus() {
		// Update the info text
		infoLabel.setText(player.generateInfoText());
		// Update the step slide
		stepSlider.setValue(player.getCurrentStep());
	}

	@Override
	public final void actionPerformed(ActionEvent e) {
		if(e.getSource()==playerTimer.playerTask) {
			// If the current state is playing, the timer's tick triggers a step
			if(state == PlayerState.PLAYING) {
				step();
			}
			return;
		}
		if(e.getSource() == playButton) {
			switch(state) {
				case STOPPED :
					stop(); break;
				case PAUSED  :
					play(); break;
				case PLAYING  :
					pause(); break;
			}
			return;
		}
		if(e.getSource() == stepButton) {
			pause();
			step();
			return;
		}
		if(e.getSource() == stopButton) {
			stop();
			return;
		}
		if(e.getSource() == prevButton || e.getSource() == nextButton) {
			stop();
			if(e.getSource() == prevButton) {
				player.previousEpisode();
			} else {
				player.nextEpisode();
			}
			// Update the number of ticks of the step slider
			if(player.getCurrentEpisode() != null) {
	        	stepSlider.setMaximum(player.getCurrentEpisode().getT());
	        } else {
	        	stepSlider.setMaximum(1);
	        }
			// Perform one step to get a display
			step();
			// If autoplay is enable, start playing the new episode
			if(autoplayBox.isSelected()) {
				play();
			}
			return;
		}
	}

	@Override
	public final void itemStateChanged(ItemEvent e) {
		Object source = e.getItemSelectable();
		if(source == loopBox) {
			loop = (e.getStateChange() == ItemEvent.SELECTED);
		}
		if(source == quitItem) {
			// nothing to do
		}
	}

	@Override
	public final void stateChanged(ChangeEvent e) {
		JSlider slider = (JSlider)e.getSource();
		if(slider == speedSlider) {
			if (!slider.getValueIsAdjusting()) {
	        	long period = (long)(((double)basePeriod) / Math.pow(2., (double)(slider.getValue()-5)));
	            if(period <= 0) {
	            	period = 1;
	            }
	            playerTimer.setPeriod(period);
	        }
		} else if(slider == stepSlider) {
			if (!slider.getValueIsAdjusting()) {
				int t = stepSlider.getValue();
				if(t != player.getCurrentStep()) {
					if(t == 0) {
						stop();
					} else {
						if(state == PlayerState.STOPPED) {
							pause();
						}
					}
					player.setStep(t);
					updateStatus();
				}
	        }
		}
	}

	@Override
	public final void dispose() {
		super.dispose();
		playerTimer.cancel();
	}

	public final class PlayerTimer extends Timer {
		private final EpisodePlayerUI player;
		private PlayerTimerTask playerTask;

		public PlayerTimer(EpisodePlayerUI player, long period) {
			this.player = player;
			playerTask = null;
			setPeriod(period);
		}

		public final void setPeriod(long period) {
			if(playerTask != null) {
				playerTask.cancel();
			}
			playerTask = new PlayerTimerTask(player);
			scheduleAtFixedRate(playerTask, 0, period);
		}
	}

	public final class PlayerTimerTask extends TimerTask {
		private final EpisodePlayerUI player;
		public PlayerTimerTask(EpisodePlayerUI player) {
			this.player = player;
		}
		@Override
		public final void run() {
			player.actionPerformed(new ActionEvent(this, 0, null));
		}
	}

	public final class OpenAction extends AbstractAction {
		public final EpisodePlayer player;
		public OpenAction(EpisodePlayer player) {
			super("Open");
			putValue(MNEMONIC_KEY, new Integer('O'));
			this.player = player;
		}
		@Override
		public final void actionPerformed(ActionEvent event) {
			int retval = fileChooser.showOpenDialog(EpisodePlayerUI.this);
            if (retval == JFileChooser.APPROVE_OPTION) {
        		try {
					Episode epi = Episode.readFromBinaryFile(
							fileChooser.getSelectedFile());
					player.addEpisode(epi);
					updateStatus();
				} catch (Exception ex) {
					JOptionPane.showMessageDialog(EpisodePlayerUI.this, ex);
				}
            }
		}
	}

	public final class OpenInNewWindowAction extends AbstractAction {
		public OpenInNewWindowAction() {
			super("Open in new window");
			putValue(MNEMONIC_KEY, new Integer('N'));
		}
		@Override
		public final void actionPerformed(ActionEvent event) {
			int retval = fileChooser.showOpenDialog(EpisodePlayerUI.this);
            if (retval == JFileChooser.APPROVE_OPTION) {
				try {
					final Episode epi = Episode.readFromBinaryFile(
							fileChooser.getSelectedFile());
					final Iterable<Episode> epis = Arrays.asList(new Episode[]{epi});
					new EpisodePlayerUI(title, disp, epis, basePeriod, gamma);
				} catch (Exception e) {
					// FIXME should do something about it
				}
            }
		}
	}

	public final class SaveAsAction extends AbstractAction {
		public final EpisodePlayer player;
		public SaveAsAction(EpisodePlayer player) {
			super("Save as");
			putValue(MNEMONIC_KEY, new Integer('S'));
			this.player = player;
		}
		@Override
		public final void actionPerformed(ActionEvent event) {
			int retval = fileChooser.showSaveDialog(EpisodePlayerUI.this);
            if (retval == JFileChooser.APPROVE_OPTION) {
        		try {
					player.getCurrentEpisode().writeToBinaryFile(
							fileChooser.getSelectedFile());
				} catch (Exception ex) {
					JOptionPane.showMessageDialog(EpisodePlayerUI.this, ex);
				}
            }
		}
	}

	public final class CloseAction extends AbstractAction {
		public final EpisodePlayer player;
		public CloseAction(EpisodePlayer player) {
			super("Close");
			putValue(MNEMONIC_KEY, new Integer('W'));
			this.player = player;
		}
		@Override
		public final void actionPerformed(ActionEvent event) {
			player.removeCurrentEpisode();
			updateStatus();
		}
	}

	public final class QuitAction extends AbstractAction {
		public QuitAction() {
			super("Quit");
			putValue(MNEMONIC_KEY, new Integer('Q'));
		}
		@Override
		public final void actionPerformed(ActionEvent e) {
			dispose();
		}
	}

	public final class ShowStatsAction extends AbstractAction {
		public ShowStatsAction() {
			super("Show stats");
			putValue(MNEMONIC_KEY, new Integer('T'));
		}
		@Override
		public final void actionPerformed(ActionEvent event) {
			new EpisodeStatsUI(episodes, gamma);
		}
	}
}
