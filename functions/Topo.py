import pandas as pd 
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

class Topo():
    def __init__(self, Sig):
        '''
        Sig: R^{nx3} including X, Y cordinate of electrode + signal strength
        n : number of channel
        '''
        self.Sig = Sig
        self.gridz = None
        self.points = None
        self.values = None
        self.grid_x = None
        self.grid_y = None

    def _mgrid(self, padding=.1, resolution = 1000j, _method = 'cubic'):
        
        x_min, x_max, y_min, y_max = np.min(self.Sig[:,0]), np.max(self.Sig[:,0]), \
            np.min(self.Sig[:,1]), np.max(self.Sig[:,1]), 
        
        dimX = x_max - x_min
        dimY = y_max - y_min

        x_min = x_min - dimX * padding
        x_max = x_max + dimX * padding
        y_min = y_min - dimY * padding
        y_max = y_max + dimY * padding


        grid_x, grid_y = np.mgrid[x_min:x_max:resolution, \
            y_min:y_max:resolution]

        points = self.Sig[:, 0:2]
        values = self.Sig[:,2]

        grid_z = griddata(points, values, (grid_x, grid_y), method=_method)

        self.gridz = grid_z.T
        self.points = points
        self.values = values,
        self.bound = (x_min, x_max, y_min, y_max)
        self.sigs = values

        return self.gridz, self.bound, self.points 

    def plot(self):
        # Interpolate data
        self._mgrid()
        plt.set_cmap('Reds')
        
        # plt.subplots()
        
        plt.imshow(self.gridz, extent = self.bound, alpha = .5, \
             origin='lower') #Important, else show inverse image
        

        # plt.contourf(self.gridz, extent = self.bound, alpha = .75)
        cs = plt.contour(self.gridz, extent = self.bound, alpha = 1, \
            origin='lower')
        
        plt.clabel(cs, inline=1)

        plt.scatter(self.points[:,0],self.points[:,1], s=2, c='black')
        
        x_min, x_max, y_min, y_max = self.bound
        
        plt.text((x_min+x_max)/2, y_max, "Front", horizontalalignment='center')
        # plt.text(x_min, (y_min+y_max)/2, "Left")
        # plt.text(x_max, (y_min+y_max)/2, "Right")
        plt.axis('off')
        plt.show()


__all__ = ['Topo']

if __name__ == "__main__":
    df = pd.read_csv(_URL)
    Sig = df[['x','y', 'signal']].values

    T = Topo(Sig)
    T.plot()
    # plt.show()


