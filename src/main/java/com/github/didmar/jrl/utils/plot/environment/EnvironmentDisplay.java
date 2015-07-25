package com.github.didmar.jrl.utils.plot.environment;

import java.awt.Color;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.event.MouseInputListener;

import com.github.didmar.jrl.environment.Environment;
import com.github.didmar.jrl.environment.EnvironmentListener;

// TODO ajouter la possibilité de déplacer la vue à la souris

/**
 * Abstract class to display an {@link Environment} using a {@link JPanel}.
 * @author Didier Marin
 */
public abstract class EnvironmentDisplay extends JPanel implements
		EnvironmentListener, MouseWheelListener, MouseMotionListener,
		MouseInputListener {

	private static final long serialVersionUID = 113929234896324126L;

	protected double zoom;
	private final double defaultZoom;
	protected int xOffset = 0;
	protected int yOffset = 0;
	private int x;
	private int y;

	public EnvironmentDisplay(double zoom) {
		setZoom(zoom);
		this.defaultZoom = zoom;
		addMouseWheelListener(this);
		addMouseMotionListener(this);
		addMouseListener(this);
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseMotionListener#mouseDragged(java.awt.event.MouseEvent)
	 */
	@Override
	public final void mouseDragged(MouseEvent e) {
		int dx = e.getX() - x;
		int dy = e.getY() - y;
		xOffset += dx;
		yOffset += dy;
		x = e.getX();
		y = e.getY();
		repaint();
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseMotionListener#mouseMoved(java.awt.event.MouseEvent)
	 */
	@Override
	public final void mouseMoved(MouseEvent e) {
		// Nothing to do
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mouseClicked(java.awt.event.MouseEvent)
	 */
	@Override
	public final void mouseClicked(MouseEvent e) {
		if(e.getButton() == MouseEvent.BUTTON3) {
			// Reset offset
			xOffset = 0;
			yOffset = 0;
			repaint();
		} else if (e.getButton() == MouseEvent.BUTTON2) {
			// Reset zoom
			setZoom( defaultZoom );
			repaint();
		}
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mouseEntered(java.awt.event.MouseEvent)
	 */
	@Override
	public final void mouseEntered(MouseEvent e) {
		// Nothing to do
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mouseExited(java.awt.event.MouseEvent)
	 */
	@Override
	public final void mouseExited(MouseEvent e) {
		// Nothing to do
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mousePressed(java.awt.event.MouseEvent)
	 */
	@Override
	public final void mousePressed(MouseEvent e) {
		x = e.getX();
		y = e.getY();
	}

	/* (non-Javadoc)
	 * @see java.awt.event.MouseListener#mouseReleased(java.awt.event.MouseEvent)
	 */
	@Override
	public final void mouseReleased(MouseEvent e) {
		// Nothing to do
	}

	public final void setZoom(double zoom) {
		if(zoom <= 0.) {
			this.zoom = 1e-5;
		} else {
			this.zoom = zoom;
		}
	}

	public final JFrame openInJFrame(int width, int height, String title) {
		JFrame frame = new JFrame(title);
		frame.setBackground(Color.WHITE);
		this.setBackground(Color.WHITE);
		frame.setSize(width, height);

		frame.setContentPane(this);

		//BorderLayout l = new BorderLayout();
		//l.addLayoutComponent(this, BorderLayout.CENTER);
		//JSlider scaleSlider = new JSlider();
		//scaleSlider.addChangeListener(this);
		//l.addLayoutComponent(scaleSlider,BorderLayout.SOUTH);
		//frame.setLayout(l);
		//this.setVisible(true);

		frame.setVisible(true);
		return(frame);
	}

	@Override
	public final void mouseWheelMoved(MouseWheelEvent e) {
		if(e.getWheelRotation() > 0) {
			setZoom(zoom / 2.);
		} else {
			setZoom(zoom * 2.);
		}
		repaint(); // FIXME does not work if player is stopped/paused !!
	}


}
