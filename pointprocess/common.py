import os
import pandas as pd

def to_decimal(degree, minute, second):
    return(degree+(minute/60.)+(second/3600.))

def fix_t(t, base):
    t = pd.Timestamp(t)
    if t.hour != base:
        t += pd.DateOffset(hours=base)
    return(t)

def import_r_tools(filename='r-tools.R'):
    from rpy2.robjects import pandas2ri, r, globalenv
    from rpy2.robjects.packages import STAP
    pandas2ri.activate()
    path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(path,filename), 'r') as f:
        string = f.read()
    rfuncs = STAP(string, "rfuncs")
    return rfuncs

def dotvars(**kwargs):
    res = {}
    for k, v in kwargs.items():
        res[k.replace('_', '.')] = v
    return res

def get_fsizes(fnames, tr):
    fsizes = []
    for i in range(len(fnames)):
        try:
            fsizes.append(os.stat(fnames[i]).st_size)
        except:
            fsizes.append(0)
    s = pd.Series(fsizes, index=tr)
    s = s.sort_values(ascending=False)
    return s

def smooth_grid(grid, sigma=3, **kwargs):
    from scipy.ndimage.filters import gaussian_filter
    return gaussian_filter(grid, sigma, **kwargs)

def filter_out_CC(ds, method='range', amax=0, amin=10):
    if method == 'CG':
        return ds.isel(record=((ds['cloud_ground'] == b'G') |
                               (ds['cloud_ground'] == 'G')))
    elif method == 'range':
        return ds.isel(record=((ds['amplitude']<amax) |
                               (ds['amplitude']>amin)))
    elif method == 'less_than':
        return ds.isel(record=(ds['amplitude']<amax))

def calculate_bearing(pointA, pointB):
    """
    Modified from: https://gist.github.com/jeromer/2005586
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    And the bearing in
    θ = atan2(sin(Δlong).cos(lat2),
          cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    """
    from math import radians, cos, sin, atan2, degrees
    (lat1, lon1), (lat2, lon2) = pointA, pointB
    if lon1 == lon2 and lat1 == lat2:
        return None

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1

    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(dlon))

    initial_bearing = atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing
