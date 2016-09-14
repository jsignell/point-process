import os
import numpy as np
import pandas as pd
import xarray as xr
from .plotting import plot_grid
from .common import *
from .databox import DataBox

class Region:
    '''
    Acronyms:
    FC = Flash Count
    DC = Diurnal Cycle
    MM = Mean Monthly
    '''
    def __init__(self, subsetted=True, opendap=False, **kwargs):
        if 'city' in kwargs:
            city = kwargs['city']
        else:
            city = kwargs
        self.CENTER = (city['lat'], city['lon'])
        self.RADIUS = city['r']
        self.PATH = city['path']
        self.SUBSETTED = subsetted
        self.OPENDAP = opendap

    def show(self):
        for attr in ['center', 'radius', 'path', 'subsetted']:
            if hasattr(self, attr.upper()):
                print('{a}: {val}'.format(a=attr, val=eval('self.'+attr.upper())))

    def define_grid(self, nbins=None, step=.01, extents=[], **kwargs):
        if len(extents) > 0:
            minx, maxx, miny, maxy = extents
        else:
            minx = self.CENTER[1]-self.RADIUS
            maxx = self.CENTER[1]+self.RADIUS
            miny = self.CENTER[0]-self.RADIUS
            maxy = self.CENTER[0]+self.RADIUS
        if nbins:
            self.gridx = np.linspace(minx, maxx, nbins)
            self.gridy = np.linspace(miny, maxy, nbins)
        else:
            self.gridx = np.arange(minx, maxx+step, step)
            self.gridy = np.arange(miny, maxy+step, step)

    def get_ds(self, cols=['strokes', 'amplitude'], func='grid',
               filter_CG=False, y='*', m='*', d='*'):
        '''
        Get the dataset for the region and time (assumes function is available locally)

        Parameters
        ----------
        cols: columns to include in final dataset - ['strokes', 'amplitude']
        func: functions to run on the dataset - 'grid'
        filter_CG: False or dict indicating method and args for filtering
                   method: {'CG', 'range', 'less_than'}
                   amin: amplitude which CG must exceed (eg. 40)
                   amax: amplitude which CG must be less than (eg.-10)
        y: int, or str indicating year with wildcards allowed
        m: int indicating month
        d: int indicating day

        Returns
        -------
        ds: concatenated xr.Dataset for the region and time
        '''
        if self.SUBSETTED:
            def preprocess_func(ds):
                return ds[cols]
        else:
            def preprocess_func(ds):
                lat, lon = self.CENTER
                r = self.RADIUS
                bounding_box = ((ds.lat<(lat+r)) & (ds.lat>(lat-r)) &
                                (ds.lon<(lon+r)) & (ds.lon>(lon-r)))
                return ds[cols].where(bounding_box).dropna('record')
        if self.OPENDAP:
            if y == '*':
                y_first = 1991
                y_last = pd.datetime.now().year+1
                dates = pd.date_range(str(y_first), str(y_last))
            else:
                dates = pd.date_range(str(y), str(y+1))
            if m != '*':
                dates = dates[dates.month == m]
            if d != '*':
                dates = dates[dates.day == d]
            fnames = [self.to_ncfile(date, check_existence=False)
                      for date in dates]
            ds = xr.open_mfdataset(fnames,
                                   concat_dim='record',
                                   preprocess=preprocess_func)
        else:
            # generate wildcard notation representing
            # yyyy, mm, dd = ('*', '*', '*')
            # if 'y' in kwargs:
            #     yyyy = str(kwargs['y'])
            # if 'm' in kwargs:
            #     mm = '{mm:02d}'.format(mm=kwargs['m'])
            # if 'd' in kwargs:
            #     dd = '{dd:02d}'.format(dd=kwargs['d'])
            if m != '*':
                m = '{mm:02d}'.format(mm=m)
            if d != '*':
                d = '{dd:02d}'.format(dd=d)
            fname = '{y}_{m}_{d}.nc'.format(y=y, m=m, d=d)
            ds = xr.open_mfdataset(self.PATH+fname, concat_dim='record',
                                   preprocess=preprocess_func)
        if filter_CG:
            ds = filter_out_CC(ds, **filter_CG)
        if func == 'grid':
            self.set_x_y(ds)
            self.FC_grid = self.__to_grid()
        return ds

    def set_x_y(self, ds):
        self.x = ds.lon.values
        self.y = ds.lat.values

    def __to_grid(self, group=None, **kwargs):
        if hasattr(group, '__iter__'):
            grid, _, _ = np.histogram2d(self.x[group], self.y[group], bins=[self.gridx, self.gridy], **kwargs)
        else:
            grid, _, _ = np.histogram2d(self.x, self.y, bins=[self.gridx, self.gridy], **kwargs)
        return grid.T

    def to_DC_grid(self, ds):
        '''
        Count the number of lightning strikes in each grid cell at each hour of the day

        Parameters
        ----------
        ds: xr.Dataset with time, lat, and lon as coordinates

        Returns
        -------
        self.DC_grid: dictionary of gridded FC for each hour of the day
        '''
        if not hasattr(self, 'x'):
            self.set_x_y(ds)
        gb = ds.groupby('time.hour')
        d = {}
        for k, v in gb.groups.items():
            d.update({k: self.__to_grid(v)})
        self.DC_grid = d

    def to_ncfile(self, t, check_existence=True, full_path=True):
        t = pd.Timestamp(t)
        fname = str(t.date()).replace('-','_')+'.nc'
        if check_existence:
            if not os.path.isfile(self.PATH+fname):
                return
        if full_path:
            return self.PATH+fname
        return fname

    def get_daily_ds(self, t, base=12, func='grid', filter_CG=False):
        '''
        Get the dataset for the region and day using the base hour
        (assumes function is available locally).
        If you are interested in 0to0 days it is equivalent to using:
        xr.open_dataset(self.to_ncfile(t))

        Parameters
        ----------
        t: str or pd.Timestamp indicating date
        base: int indicating hours between which to take day - 12
        func: functions to run on the dataset - 'grid'
        filter_CG: False or dict indicating method and args for filtering
                   method: {'CG', 'range', 'less_than'}
                   amin: amplitude which CG must exceed (eg. 40)
                   amax: amplitude which CG must be less than (eg.-10)

        Returns
        -------
        ds: concatenated xr.Dataset for the region and day
        '''
        t = fix_t(t, base)
        if base == 0:
            ds0 = xr.open_dataset(self.to_ncfile(t))
        else:
            L = list(filter(None, [self.to_ncfile(day) for
                                   day in [t, t+pd.DateOffset(1)]]))
            if len(L)==0:
                if func=='count':
                    return 0
                return
            ds = xr.concat([xr.open_dataset(f) for f in L], dim='record')
            UTC12 = [np.datetime64(day) for day in pd.date_range(start=t, periods=2)]

            ds0 = ds.where((ds.time>UTC12[0]) & (ds.time<UTC12[1])).dropna('record')
            if filter_CG:
                ds0 = filter_out_CC(ds0, **filter_CG)
            ds.close()

        if func=='grid':
            self.set_x_y(ds0)
            self.FC_grid = self.__to_grid()

        if func=='count':
            count = ds0.record.size
            ds0.close()
            return(count)

        if func=='max':
            self.set_x_y(ds0)
            self.FC_grid = self.__to_grid()
            ds0.close()
            return(self.FC_grid.max())

        return(ds0)

    def area_over_thresh(self, thresh=[1,2,5], print_out=True, return_dict=False):
        area = {}
        for n in thresh:
            area.update({n: (self.FC_grid>=n).sum()})
        if print_out:
            print('\n'.join(['Area exceeding {thresh} strikes: {area} km^2'.format(thresh=n, area=area[n]) for n in thresh]))
        if return_dict:
            return(area)

    def get_daily_grid_slices(self, t, base=12, **kwargs):
        '''
        For the pre-defined grid, use indicated frequency to also bin along the time dimension

        Parameter
        --------
        t: str or pd.Timestamp indicating date
        base: int indicating hours between which to take day - 12
        freq: str indicating frequency as in pandas - '5min'
        filter_CG: bool indicating whether or not to take only cloud to ground events
                For new style events uses 'C', 'G' flag, otherwise use strokes where
                amplitude > 10 or amplitude < 0
                (after: Cummins et al. 1998 and Orville et al. 2002)

        Returns
        -------
        box: np.array of shape (ntimesteps, ny, nx)
        tr: timerange of shape (ntimesteps)

        Benchmarking
        ------------
        4.26 s for 600x600 1min
        1.16 s for 600x600 5min
        608 ms for 60x60 5min
        '''
        ds = self.get_daily_ds(t, base=base, func=None, **kwargs)
        t = fix_t(t, base)
        start = t
        end = t+pd.DateOffset(1)
        box, tr = self.get_grid_slices(ds, start, end, **kwargs)
        ds.close()
        return(box, tr)

    def get_grid_slices(self, ds, start=None, end=None, freq='5min', tr=None, filter_CG=False):
        '''
        For the pre-defined grid, use indicated frequency to also bin along the time dimension

        Parameter
        --------
        ds: xr.dataset for a short amount of time (a couple days)
        start: str or pd.Timestamp indicating start time for slices
        end: str or pd.Timestamp indicating end time for slices
        freq: str indicating frequency as in pandas - '5min'
        filter_CG: bool indicating whether or not to take only cloud to ground events
                For new style events uses 'C', 'G' flag, otherwise use strokes where
                amplitude > 10 or amplitude < 0
                (after: Cummins et al. 1998 and Orville et al. 2002)

        Returns
        -------
        box: np.array of shape (ntimesteps, ny, nx)
        tr: timerange of shape (ntimesteps)
        '''
        if filter_CG:
            ds = filter_out_CC(ds, **filter_CG)
        df = ds.to_dataframe()
        df.index = df.time
        if tr is None:
            tr = pd.date_range(start, end, freq=freq)
        d = []
        for i in range(len(tr)-1):
            grid, _,_ = np.histogram2d(df.lon[tr[i]:tr[i+1]].values,
                                       df.lat[tr[i]:tr[i+1]].values,
                                       bins=[self.gridx, self.gridy])
            d.append(grid.T)
        return(np.stack(d), tr)

    def get_top(self, n=100, base=12, year=None):
        '''
        Quick and dirty method for finding top n FC days for a subsetted
        region. This method uses file size as a proxy for number of events
        and sorts out the n*2 largest of the old style files and the largest
        n*2 of the new style files. It is not proven to work well for n<10.

        Parameters
        ----------
        n: number of top events
        base: hour of day to start and end daily on
        year: choose once particular year to check for largest days

        Returns
        -------
        s: sorted series containing days and lightning event counts for top
        n days
        '''
        if not self.SUBSETTED:
            print('This method only works for pre-subsetted regions.')
            return
        d={}
        if year is not None:
            start = '{y}-01-01'.format(y=year)
            end = '{y}-12-31'.format(y=year)
            d = self.__get_big(start, end, d, n, base)
        else:
            for start, end in [('1991-01-01', '2010-01-01'),
                               ('2010-01-01', '2015-10-02')]:
                d = self.__get_big(start, end, d, n, base)
        s = pd.Series(d).sort_values(ascending=False).head(n)
        return s

    def __get_big(self, start, end, d, n, base):
        tr = pd.date_range(fix_t(start, base), fix_t(end, base))
        fnames = [self.to_ncfile(t, check_existence=False) for t in tr]
        s = get_fsizes(fnames, tr)
        for i in range(n):
            little_tr = pd.date_range(start=s.index[i]-pd.DateOffset(1),
                                      periods=3)
            little_fnames = list(filter(None, [self.to_ncfile(t) for
                                               t in little_tr]))
            ds = xr.concat([xr.open_dataset(f) for f in little_fnames],
                           dim='record')
            UTC12 = [np.datetime64(t) for t in little_tr]
            first = ds.where((ds.time>UTC12[0]) &
                             (ds.time<UTC12[1])).dropna('record').record.shape[0]
            second = ds.where((ds.time>UTC12[1]) &
                             (ds.time<UTC12[2])).dropna('record').record.shape[0]
            d.update({pd.Timestamp(UTC12[0]): first})
            d.update({pd.Timestamp(UTC12[1]): second})
            ds.close()
        return d

    def plot_grid(self, grid, **kwargs):
        '''
        Simple and fast plot generation for gridded data

        Parameters
        ----------
        grid: np.array with shape matching gridx by gridy
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
        return(plot_grid(self.gridy, self.gridx, grid, **kwargs))

    def to_databox(self, box, tr):
        lon, lat = np.meshgrid(self.gridx[0:-1], self.gridy[0:-1])
        return(DataBox(tr, lat, lon, box))

    # def add_buffer(self, p):
    #     from geopy.distance import vincenty

    #     edges = zip([self.gridy[0]]*100, np.linspace(self.gridx[0], self.gridx[-1],100))
    #     edges.extend(zip(self.linspace(self.gridy[0], self.gridy[-1],100), [self.gridx[-1]]*100))
    #     edges.extend(zip([self.gridy[-1]]*100, np.linspace(self.gridx[0], self.gridx[-1],100)))
    #     edges.extend(zip(np.linspace(self.gridy[0], self.gridy[-1],100), [self.gridx[0]]*100))

    #     for it in range(p.shape[0]):
    #         for ifeat in range(p.shape[1]):
    #             if np.isnan(p[it, ifeat, 'centroidY']):
    #                 continue
    #             center = p[it, ifeat, ['centroidY', 'centroidX']].values
    #             dist = min([vincenty(center, edge).kilometers for edge in edges])
    #             r = (p[it, ifeat, ['area']].values/np.pi)**.5
    #             if r>dist:
    #                 df0 = p[it,:,:]
    #                 for ichar in range(21):
    #                     df0.set_value(p.major_axis[ifeat], p.minor_axis[ichar], np.nan)
    #     return(p)

    # def get_features(self, box, tr, thresh=.01, sigma=3, min_size=4, const=5, buffer=False):
    #     '''
    #     Use r package SpatialVx to identify features.

    #     Parameters
    #     ----------
    #     box: grid slices as returned from self.get_grid_slice()
    #     '''
    #     from rpy2 import robjects
    #     from rpy2.robjects.packages import importr
    #     from rpy2.robjects import pandas2ri
    #     pandas2ri.activate()
    #     SpatialVx = importr('SpatialVx')
    #     rsummary = robjects.r.summary
    #     r_tools = import_r_tools()

    #     d = {}
    #     X, Y = np.meshgrid(self.gridx[0:-1], self.gridy[0:-1])
    #     ll = np.array([X.flatten('F'), Y.flatten('F')]).T
    #     for i in range(box.shape[0]-1):
    #         hold = SpatialVx.make_SpatialVx(box[i,:,:], box[i+1,:,:], loc=ll)
    #         look = r_tools.FeatureFinder_gaussian(hold, nx=box.shape[2], ny=box.shape[1],
    #                                               thresh=thresh, smoothpar=sigma, **(dotvars(min_size=min_size)))
    #         try:
    #             x = rsummary(look, silent=True)[0]
    #         except:
    #             continue
    #         px = pandas2ri.ri2py(x)
    #         df0 = pd.DataFrame(px, columns=['centroidX', 'centroidY', 'area', 'OrientationAngle',
    #                                       'AspectRatio', 'Intensity0.25', 'Intensity0.9'])
    #         df0['Observed'] = list(df0.index+1)
    #         m = SpatialVx.centmatch(look, criteria=3, const=const)
    #         p = pandas2ri.ri2py(m[12])
    #         df1 = pd.DataFrame(p, columns=['Forecast', 'Observed'])
    #         l = SpatialVx.FeatureMatchAnalyzer(m)
    #         try:
    #             p = pandas2ri.ri2py(rsummary(l, silent=True))
    #         except:
    #             continue
    #         df2 = pd.DataFrame(p, columns=['Partial Hausdorff Distance','Mean Error Distance','Mean Square Error Distance',
    #                                       'Pratts Figure of Merit','Minimum Separation Distance', 'Centroid Distance',
    #                                       'Angle Difference','Area Ratio','Intersection Area','Bearing', 'Baddeleys Delta Metric',
    #                                       'Hausdorff Distance'])
    #         df3 = df1.join(df2)

    #         d.update({tr[i]: pd.merge(df0, df3, how='outer')})
    #     p = pd.Panel(d)
    #     if buffer:
    #         return(self.add_buffer(p))
    #     return(p)

    # def kde(self, lon, lat):
    #     import scipy.stats as st
    #     xx, yy = self.gridx, self.gridy
    #     positions = np.vstack([xx.ravel(), yy.ravel()])
    #     values = np.vstack([lon, lat])
    #     kernel = st.gaussian_kde(values)
    #     f = np.reshape(kernel(positions).T, xx.shape)
    #     return(xx,yy,f)
