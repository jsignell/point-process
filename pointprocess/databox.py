import numpy as np
import pandas as pd
from .common import *
from .plotting import plot_grid

class DataBox:
    def __init__(self, time, lat, lon, box):
        self.nt, self.ny, self.nx = box.shape
        if time.shape == self.nt or len(time) == self.nt:
            self.time = time
            self.t_edge = False
        elif time.shape == (self.nt+1) or len(time) == (self.nt+1):
            self.time = time
            self.t_edge = True
        else:
            print('check time')
        if lat.shape == (self.ny, self.nx):
            self.dims = 2
        elif lat.shape == (self.ny):
            self.dims = 1
            self.l_edge = False
            self.get_X()
        elif lat.shape == (self.ny+1):
            self.dims = 1
            self.l_edge = True
            self.get_X()
        else:
            print('check lat, lon')
        self.lat = lat
        self.lon = lon
        self.box = box

    def get_X(self):
        if self.dims == 2:
            return
        if self.l_edge:
            xx, yy = np.meshgrid(self.lon[1:], self.lat[1:])
        else:
            xx, yy = np.meshgrid(self.lon, self.lat)
        xx = xx.flatten()
        yy = yy.flatten()
        self.X = np.stack([yy, xx])

    def show(self):
        print('DataBox of shape: {shape}'.format(shape=self.box.shape))

    def get_l(self, x):
        '''
        Given a latlon position, return the 1D and 2D index
        of the corresponding grid cell. Also works on series
        of positions

        Parameters
        ----------
        x: (lat, lon) where lat, lon are floats or series
        grid_lat: list of the lat edges of the y grid cells
        grid_lon: list of the lon edges of the x grid cells

        Returns
        -------
        yloc, xloc, l: tuple of integers or tuple of series

        Examples
        --------
        # for one position
        db.get_l((df.lat[0], df.lon[0]))

        df.assign(**dict(list(zip(['yloc', 'xloc', 'l'],
                                  db.get_l((df.lat, df.lon))
        '''
        if self.dims == 2:
            return
        def get_loc(ll, grid_ll):
            if ll<grid_ll[-1] and ll>grid_ll[0]:
                return np.argmax(grid_ll>ll)
            else:
                return np.nan
        lat, lon = x
        if hasattr(lat, '__iter__'):
            yloc = lat.apply(get_loc, grid_ll=self.lat)
            xloc = lon.apply(get_loc, grid_ll=self.lon)
        else:
            yloc = get_loc(lat, self.lat)
            xloc = get_loc(lon, self.lat)
        l = (yloc-1)*(grid_lon.shape[0]-1)+(xloc-1)
        return yloc, xloc, l

    def flat_plot(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))

        axes[0].plot(np.sum(self.box, axis=(1,2)))
        axes[0].set_title("Flattened t axis");

        axes[1].plot(np.sum(self.box, axis=(0,2)))
        axes[1].set_title("Flattened y axis")

        axes[2].plot(np.sum(self.box, axis=(0,1)))
        axes[2].set_title("Flattened x axis")
        return(fig)

    def plot_grid(self, grid=None, **kwargs):
        '''
        Simple and fast plot generation for gridded data

        Parameters
        ----------
        grid: np.array with shape matching self.lat
        ax: matplotlib axes object, if not given generates and populates with basic map
        cbar: bool indicating whether or not to show default colorbar
        **kwargs: fed directly into ax.imshow()

        Returns
        -------
        im, ax: (output from ax.imshow, matplotlib axes object)

        Benchmarking
        ------------
        33.8 ms for 600x600
        32.9 ms for 60x60
        '''
        if grid is None:
            grid = np.nansum(self.box, axis=0)
        return(plot_grid(self.lat, self.lon, grid, **kwargs))

    def get_gauss2d(self, sigma=3):
        from scipy.ndimage.filters import gaussian_filter
        gauss2d = np.array([gaussian_filter(self.box[i,:,:], sigma) for
                            i in range(self.box.shape[0])])
        return(gauss2d)

    def centralized_difference(self, t_start=None, t_end=None,
                               radius=15, buffer=20, save=False, **kwargs):
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        l =[]
        count=0
        if radius+3 < buffer:
            r=radius+3
        else:
            r=buffer
        ixy0 = buffer
        ixyn = self.lat.shape[1]-buffer
        try:
            it0 = self.time.get_loc(t_start)
            itn = self.time.get_loc(t_end)
        except:
            it0 = 0
            itn = self.time.shape[0]-2
        for ix in range(ixy0, ixyn):
            for iy in range(ixy0, ixyn):
                for it in range(it0, itn):
                    here = self.box[it+1, iy-r:iy+r+1, ix-r:ix+r+1]-self.box[it, iy, ix]
                    if not np.isnan(np.sum(here)):
                        if count == 0:
                            test = here
                            count+=1
                        else:
                            test += here
                            count+=1
        test/=float(count)
        if 'vmin' not in kwargs.keys():
            peak = max(np.abs(np.min(test[r-radius:radius+r, r-radius:radius+r])),
                       np.max(test[r-radius:radius+r, r-radius:radius+r]))
            kwargs.update(dict(vmin = -peak, vmax = peak))
        if 'nrows' in kwargs.keys():
            nrows = kwargs.pop('nrows')
            ncols = kwargs.pop('ncols')
            n = kwargs.pop('n')
        else:
            nrows = ncols = n = 1
        ax = plt.subplot(nrows, ncols, n, projection=ccrs.PlateCarree())

        scat = ax.pcolor(self.lon[iy-r:iy+r+1, ix-r:ix+r+1], self.lat[iy-r:iy+r+1, ix-r:ix+r+1], test, **kwargs)
        ax.set_extent([self.lon[iy, ix-radius], self.lon[iy, ix+radius], self.lat[iy-radius, ix], self.lat[iy+radius, ix]])
        ax.scatter(self.lon[iy, ix], self.lat[iy, ix], edgecolor='white', facecolor='None')
        return(scat, ax, kwargs['vmax'])

    def add_buffer(self, p, extra=0, aspect=False):
        from geopy.distance import vincenty

        edges = list(zip(self.lat[0, :], self.lon[0, :]))
        edges.extend(list(zip(self.lat[:, -1], self.lon[:, -1])))
        edges.extend(list(zip(np.flipud(self.lat[-1, :]), np.flipud(self.lon[-1, :]))))
        edges.extend(list(zip(np.flipud(self.lat[:, 0]), np.flipud(self.lon[:, 0]))))

        for it in range(p.shape[0]):
            for ifeat in range(p.shape[1]):
                if np.isnan(p[it, ifeat, 'centroidY']):
                    continue
                center = p[it, ifeat, ['centroidY', 'centroidX']].values
                dist = min([vincenty(center, edge).kilometers for edge in edges])
                r = (p[it, ifeat, ['area']].values/np.pi)**.5
                if aspect:
                    asp = p[it, ifeat,['AspectRatio']].values
                    if asp>1:
                        asp=1/asp
                else:
                    asp = 1
                if ((r+extra)/asp)>dist:
                    df0 = p[it,:,:]
                    for ichar in range(21):
                        df0.set_value(p.major_axis[ifeat], p.minor_axis[ichar], np.nan)
        return p

    def get_features(self, d={}, thresh=.01, sigma=3, min_size=4, const=5, return_dict=False, buffer=False):
        '''
        Use r package SpatialVx to identify features.

        Parameters
        ----------
        thresh: .01
        sigma: 3
        min_size: 4
        const: 5
        buffer: False

        Return
        ------
        p: pd.Panel containing parameters characterizing the features found
        '''
        from rpy2 import robjects
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        SpatialVx = importr('SpatialVx')
        rsummary = robjects.r.summary
        r_tools = import_r_tools()

        ll = np.array([self.lon.flatten('F'), self.lat.flatten('F')]).T
        for i in range(self.box.shape[0]-1):
            hold = SpatialVx.make_SpatialVx(self.box[i,:,:], self.box[i+1,:,:], loc=ll)
            look = r_tools.FeatureFinder_gaussian(hold, nx=self.box.shape[2], ny=self.box.shape[1],
                                                  thresh=thresh, smoothpar=sigma, **(dotvars(min_size=min_size)))
            try:
                x = rsummary(look, silent=True)[0]
            except:
                continue
            px = pandas2ri.ri2py(x)
            df0 = pd.DataFrame(px, columns=['centroidX', 'centroidY', 'area', 'OrientationAngle',
                                          'AspectRatio', 'Intensity0.25', 'Intensity0.9'])
            df0['Observed'] = list(df0.index+1)
            m = SpatialVx.centmatch(look, criteria=3, const=const)
            p = pandas2ri.ri2py(m[12])
            df1 = pd.DataFrame(p, columns=['Forecast', 'Observed'])
            l = SpatialVx.FeatureMatchAnalyzer(m)
            try:
                p = pandas2ri.ri2py(rsummary(l, silent=True))
            except:
                continue
            df2 = pd.DataFrame(p, columns=['Partial Hausdorff Distance','Mean Error Distance','Mean Square Error Distance',
                                          'Pratts Figure of Merit','Minimum Separation Distance', 'Centroid Distance',
                                          'Angle Difference','Area Ratio','Intersection Area','Bearing', 'Baddeleys Delta Metric',
                                          'Hausdorff Distance'])
            df3 = df1.join(df2)

            d.update({self.time[i]: pd.merge(df0, df3, how='outer')})
        if return_dict:
            return(d)
        p = pd.Panel(d)
        if buffer:
            return(self.add_buffer(p))
        return(p)

    def kde(self, lon, lat):
        import scipy.stats as st
        xx, yy = self.lon, self.lat
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([lon, lat])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        return(xx,yy,f)
