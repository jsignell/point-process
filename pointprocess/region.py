import os
import numpy as np
import pandas as pd
import xarray as xr
from .plotting import plot_grid, plot_contour
from .common import *
from .databox import DataBox


class Region:
    '''
    Acronyms:
    FC = Flash Count
    DC = Diurnal Cycle
    MM = Mean Monthly
    '''
    def __init__(self, subsetted=True, **kwargs):
        if 'city' in kwargs:
            city = kwargs['city']
        else:
            city = kwargs
        self.CENTER = (city['lat'], city['lon'])
        self.RADIUS = city['r']
        self.PATH = city['path']
        self.SUBSETTED = subsetted

    def show(self):
        for attr in ['center', 'radius', 'path', 'subsetted']:
            if hasattr(self, attr.upper()):
                print('{a}: {val}'.format(a=attr,
                                          val=eval('self.'+attr.upper())))

    def define_grid(self, nbins=None, step=.01, extents=[], units='latlon'):
        '''
        Define the grid over which to aggregate data. Specify nbins or step.

        Parameters
        ----------
        nbins: number of bins used in x and y dimensions
        step: 1D number of units in each bin -- ignored if nbins is set
        extents: [minx, maxx, miny, maxy] in latlon defining corners of grid
        units: units of step (ignored if nbins is specified)-- ['km', 'latlon']

        Returns
        -------
        sets self.gridx and self.gridy to 1D arrays of bin edges in latlon
        '''
        if len(extents) > 0:
            minx, maxx, miny, maxy = extents
        else:
            minx = self.CENTER[1]-self.RADIUS
            maxx = self.CENTER[1]+self.RADIUS
            miny = self.CENTER[0]-self.RADIUS
            maxy = self.CENTER[0]+self.RADIUS
        if units == 'km':
            from geopy.distance import vincenty
            ulc = (self.CENTER[0], minx)
            urc = (self.CENTER[0], maxx)
            llc = (miny, self.CENTER[1])
            lrc = (maxy, self.CENTER[1])
            stepx = (maxx-minx)*step/vincenty(ulc, urc).km
            stepy = (maxy-miny)*step/vincenty(ulc, llc).km
        else:
            stepx = stepy = step
        if nbins:
            self.gridx = np.linspace(minx, maxx, nbins)
            self.gridy = np.linspace(miny, maxy, nbins)
        else:
            self.gridx = np.arange(minx, maxx+stepx, stepx)
            self.gridy = np.arange(miny, maxy+stepy, stepy)

    def get_ds(self, cols=['strokes', 'amplitude'], func='grid',
               filter_CG=False, y='*', m='*', d='*'):
        '''
        Get the dataset for the region and time

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
                bounding_box = ((ds.lat < (lat + r)) & (ds.lat > (lat - r)) &
                                (ds.lon < (lon + r)) & (ds.lon > (lon - r)))
                return ds[cols].isel(record=bounding_box)
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
            self.__set_x_y(ds)
            self.FC_grid = self.__to_grid()
        return ds

    def __set_x_y(self, ds):
        self.x = ds.lon.values
        self.y = ds.lat.values

    def __to_grid(self, group=None, **kwargs):
        if hasattr(group, '__iter__'):
            grid, _, _ = np.histogram2d(self.x[group], self.y[group],
                                        bins=[self.gridx, self.gridy],
                                        **kwargs)
        else:
            grid, _, _ = np.histogram2d(self.x, self.y,
                                        bins=[self.gridx, self.gridy],
                                        **kwargs)
        return grid.T

    def to_DC_grid(self, ds):
        '''
        Count the number of lightning strikes in each grid cell at each hour
        of the day

        Parameters
        ----------
        ds: xr.Dataset with time, lat, and lon as coordinates

        Returns
        -------
        self.DC_grid: dictionary of gridded FC for each hour of the day
        '''
        if not hasattr(self, 'x'):
            self.__set_x_y(ds)
        gb = ds.groupby('time.hour')
        d = {}
        for k, v in gb.groups.items():
            d.update({k: self.__to_grid(v)})
        self.DC_grid = d

    def to_ncfile(self, t, check_existence=True, full_path=True):
        t = pd.Timestamp(t)
        fname = str(t.date()).replace('-', '_')+'.nc'
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
        func: functions to run on the dataset - ['grid', 'count', 'max']
        filter_CG: False or dict indicating method and args for filtering
                   method: {'CG', 'range', 'less_than'}
                   amin: amplitude which CG must exceed (eg. 40)
                   amax: amplitude which CG must be less than (eg.-10)

        Returns
        -------
        if func == 'grid':
            ds: concatenated xr.Dataset for the region and day, and
                sets self.FC_grid
        if func == 'count':
            count: int, total flash count
        if func == 'max':
            max: int, max flash count in a grid cell
        '''
        t = fix_t(t, base)
        if base == 0:
            ds0 = xr.open_dataset(self.to_ncfile(t))
        else:
            L = list(filter(None, [self.to_ncfile(day) for
                                   day in [t, t+pd.DateOffset(1)]]))
            if len(L) == 0:
                if func == 'count':
                    return 0
                return
            ds = xr.concat([xr.open_dataset(f) for f in L], dim='record')
            UTC12 = [np.datetime64(day) for day in pd.date_range(start=t,
                                                                 periods=2)]

            ds0 = ds.isel(record=((ds.time > UTC12[0]) & (ds.time < UTC12[1])))
            if filter_CG:
                ds0 = filter_out_CC(ds0, **filter_CG)
            ds.close()

        if not self.SUBSETTED:
            lat, lon = self.CENTER
            r = self.RADIUS
            bounding_box = ((ds0.lat < (lat + r)) & (ds0.lat > (lat - r)) &
                            (ds0.lon < (lon + r)) & (ds0.lon > (lon - r)))
            ds0 = ds0.isel(record=bounding_box)

        if func == 'grid':
            self.__set_x_y(ds0)
            self.FC_grid = self.__to_grid()

        elif func == 'count':
            count = ds0.record.size
            ds0.close()
            return count

        elif func == 'max':
            self.__set_x_y(ds0)
            self.FC_grid = self.__to_grid()
            ds0.close()
            return self.FC_grid.max()

        return ds0

    def area_over_thresh(self, thresh=[1, 2, 5],
                         print_out=True, return_dict=False):
        '''
        Number of grid cells over each threshold

        Parameter
        --------
        thresh: iterable, of threholds to use
        print_out: bool, print output
        return_dict: return a dictionary of

        Returns
        -------
        if return_dict:
            dict: thresholds as keys and values ngrid cells > thresh
        '''
        area = {}
        for n in thresh:
            area.update({n: (self.FC_grid >= n).sum()})
        if print_out:
            print('\n'.join([('Area exceeding {thresh} strikes: {area} '
                              'km^2').format(thresh=n, area=area[n]) for
                             n in thresh]))
        if return_dict:
            return area

    def get_daily_grid_slices(self, t, base=12, **kwargs):
        '''
        For the pre-defined grid, use indicated frequency to also bin along
        the time dimension

        Parameter
        --------
        t: str or pd.Timestamp indicating date
        base: int indicating hours between which to take day - 12
        freq: str indicating frequency as in pandas - '5min'
        filter_CG: False or dict indicating method and args for filtering
                   method: {'CG', 'range', 'less_than'}
                   amin: amplitude which CG must exceed (eg. 40)
                   amax: amplitude which CG must be less than (eg.-10)

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
        return box, tr

    def get_grid_slices(self, ds, start=None, end=None, freq='5min',
                        tr=None, filter_CG=False):
        '''
        For the pre-defined grid, use indicated frequency to also bin along
        the time dimension

        Parameter
        --------
        ds: xr.dataset for a short amount of time (a couple days)
        start: str or pd.Timestamp indicating start time for slices
        end: str or pd.Timestamp indicating end time for slices
        freq: str indicating frequency as in pandas - '5min'
        filter_CG: False or dict indicating method and args for filtering
                   method: {'CG', 'range', 'less_than'}
                   amin: amplitude which CG must exceed (eg. 40)
                   amax: amplitude which CG must be less than (eg.-10)

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
            grid, _, _ = np.histogram2d(df.lon[tr[i]:tr[i+1]].values,
                                        df.lat[tr[i]:tr[i+1]].values,
                                        bins=[self.gridx, self.gridy])
            d.append(grid.T)
        return np.stack(d), tr

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
        d = {}
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
            first = ds.isel(record=((ds.time > UTC12[0]) &
                                    (ds.time < UTC12[1]))).record.shape[0]
            second = ds.isel(record=((ds.time > UTC12[1]) &
                                     (ds.time < UTC12[2]))).record.shape[0]
            d.update({pd.Timestamp(UTC12[0]): first})
            d.update({pd.Timestamp(UTC12[1]): second})
            ds.close()
        return d

    def plot_contour(self, grid=None, **kwargs):
        '''
        Access to plotting.plot_contour
        '''
        if grid is None:
            grid = self.FC_grid
        return plot_contour(self.gridy, self.gridx, grid, **kwargs)

    def plot_grid(self, grid=None, **kwargs):
        '''
        Access to plotting.plot_grid
        '''
        if grid is None:
            grid = self.FC_grid
        return plot_grid(self.gridy, self.gridx, grid, **kwargs)

    def to_databox(self, box, tr):
        '''
        Pass Region object to DataBox

        Parameters
        ----------
        box: np.array of shape (ntimesteps, ny, nx)
        tr: timerange of shape (ntimesteps)

        Returns:
        DataBox: with lat lon indicating centers of grid cells
        '''
        cell_centers_x = pd.Series(self.gridx).rolling(2).mean().values[1:]
        cell_centers_y = pd.Series(self.gridy).rolling(2).mean().values[1:]

        lon, lat = np.meshgrid(cell_centers_x, cell_centers_y)
        return DataBox(tr, lat, lon, box)

    def conditional_rate_of_occurrence(self, ds,
                                       t_window=pd.Timedelta(minutes=15),
                                       symmetric_t_window=False,
                                       dist_window=20, dim='speed',
                                       max_dim=200,
                                       windrose=True, **kwargs):
        '''
        Calculate the conditional rate of occurence

        Parameters
        ----------
        ds: dataset with coordinates lat, lon, and time
        t_window: pd.Timedelta of time window used to check for occurrences
        symmetric_t_window: bool indicating whether to look backwards in time
        dist_window: radius in km of window in which to check for occurrences
        dim: str dimension to pass to windrose
             -- 'speed', 'dist', 'hours', or 'minutes'
        max_dim: max dim to include in windrose can be set to None
        windrose: bool indicating whether or not to plot windrose
        **kwargs: passed on to windrose function

        Returns
        -------
        plot: windrose plot
        OR
        df: pandas.Dataframe containing bearing and dim info
        '''
        from geopy.distance import vincenty, great_circle

        zero_time = pd.Timedelta(seconds=0).asm8
        t_window = t_window.asm8

        if symmetric_t_window:
            t_window = t_window/2

        lon = ds.lon.values
        lat = ds.lat.values
        time = ds.time.values
        loc = np.stack([lat, lon]).T

        dists = []
        bearings = []
        speeds = []
        t_diffs = []
        for t in time:
            if symmetric_t_window:
                t_diff = np.abs(time-t)
            else:
                t_diff = time-t
            bool_a = np.where((zero_time < t_diff) & (t_diff < t_window))

            little_loc = np.take(loc, bool_a, axis=0)[0]
            little_t_diff = np.take(t_diff, bool_a)[0]
            for ll, tt in zip(little_loc[1:], little_t_diff[1:]):
                if (ll == little_loc[0]).all():
                    continue
                dist = great_circle(little_loc[0], ll).km
                if dist < dist_window:
                    hours = (int(tt)/10e8/60/60.)
                    t_diffs.append(hours)
                    dists.append(dist)
                    speeds.append(dist/hours)
                    bearings.append(calculate_bearing(little_loc[0], ll))
        df = pd.DataFrame({'speed': speeds, 'dist': dists, 'hours': t_diffs,
                           'direction': bearings})
        if dim == 'minutes':
            df = df.assign(minutes=df.hours*60)
        if not windrose:
            return df
        else:
            from windrose import WindroseAxes

            if max_dim is not None:
                df = df[df[dim].abs() < max_dim]
            max_dim = max_dim or df[dim].abs().max()

            ax = WindroseAxes.from_ax()
            ax.bar(df['direction'], df[dim],
                   bins=kwargs.pop('bins', np.arange(0, max_dim, 20)),
                   normed=kwargs.pop('normed', True),
                   opening=kwargs.pop('opening', 0.9),
                   edgecolor=kwargs.pop('edgecolor', 'white'),
                   **kwargs)
            ax.set_legend()

            return ax
