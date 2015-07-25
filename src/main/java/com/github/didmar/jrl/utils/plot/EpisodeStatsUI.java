package com.github.didmar.jrl.utils.plot;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableModel;
import javax.swing.table.TableRowSorter;

import com.github.didmar.jrl.utils.DiscountFactor;
import com.github.didmar.jrl.utils.Episode;
import com.github.didmar.jrl.utils.array.ArrUtils;

/**
 * A user interface that displays stats about some given episodes.
 * @author Didier Marin
 */
public class EpisodeStatsUI extends JFrame {

	private static final long serialVersionUID = 1L;
	
	private static final Object[] columnNames = new Object[]{
		"Epi n°","T","J","min r","max r"
	};
	
	private static final int EPI_NUM=0, DURATION=1, PERF=2, MIN_REWARD=3, MAX_REWARD=4;

	private final JTable table;
	
	@SuppressWarnings("serial")
	public EpisodeStatsUI(Iterable<Episode> episodes, DiscountFactor gamma) {
		super("Episode stats");
		
		List<Episode> epis = new ArrayList<Episode>();
		if(episodes != null) {
			for(Episode e : episodes) {
				epis.add(e);
			}
		}
		// Create an array that contains the table data
		Object[][] data = new Object[epis.size()][columnNames.length];
		for (int i = 0; i < data.length; i++) {
			data[i][EPI_NUM] = (double)i+1;
			Episode e = epis.get(i);
			data[i][DURATION] = (double)e.getT();
			data[i][PERF] = e.discountedReward(gamma);
			double[] R = e.getR();
			data[i][MIN_REWARD] = ArrUtils.min(R);
			data[i][MAX_REWARD] = ArrUtils.max(R);
		}
		// Create a table model
		TableModel model = new DefaultTableModel(data, columnNames) {
			public Class<?> getColumnClass(int column) {
				return getValueAt(0, column).getClass();
			}
		};
		// Create a JTable from this model
		table = new JTable(model);
		table.setEnabled(false);
		// Make it sortable
		TableRowSorter<TableModel> sorter = new TableRowSorter<TableModel>(model);
	    table.setRowSorter(sorter);
	    // Create a JScrollPane to scroll through the table
		JScrollPane scrollPane = new JScrollPane(table);
		table.setFillsViewportHeight(true);
		
		Chart chart = new Chart(data);
		
		// Change the layout to BorderLayout
		setLayout(new BorderLayout());
		// Add the components to the content pane
		Container pane = getContentPane();
		pane.add(scrollPane, BorderLayout.CENTER);
		pane.add(chart, BorderLayout.EAST);
		
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setSize(640,480);
		setVisible(true);
	}
	
	@SuppressWarnings("serial")
	public class Chart extends ChartPanel {
		private Object[][] data;
		public Chart(Object[][] data) {
			this.data = data;
			setColumn(PERF);
			setPreferredSize(new Dimension(640,480));
			setVisible(true);
		}
		public void setColumn(int column) {
			double[] v =  new double[data[0].length];
			String[] n = new String[data[0].length];
			for (int i = 0; i < data[0].length; i++) {
				v[i] = (Double)(data[i][column]);
			}
			Arrays.sort(v);
			for (int i = 0; i < v.length; i++) {
				n[i] = Double.toString(v[i]);
			}
			setChart(v, n, (String)columnNames[column]);
		}
	}
	
}
