import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import pandas as pd


points = np.random.rand(10,2) #random

df = pd.read_csv('bikesharing/raw_data/2013_08_NYC.csv').head(100)

pointz = [df['start station longitude'], df['start station latitude']]

vor = Voronoi(points)

fig = voronoi_plot_2d(vor)

fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                      line_width=2, line_alpha=0.6, point_size=2)
plt.show()