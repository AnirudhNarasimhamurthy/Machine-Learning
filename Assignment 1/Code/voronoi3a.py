from scipy.spatial import Voronoi,voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt

points = np.array([[1,1],[-1,-1],[2,-2],[4,0]])
vor = Voronoi(points)
voronoi_plot_2d(vor)
plt.show()
