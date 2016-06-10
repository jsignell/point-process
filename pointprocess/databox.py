import numpy as np
import pandas as pd
from common import *

class DataBox:
    def __init__(time, lat, lon, box):
    	nt, ny, nx = box.shape
    	if time.shape == nt:
    		self.time = time
    	if lat.shape == ny:
    		self.lon, self.lat = np.meshgrid(lon, lat)
    	elif lat.shape == (ny, nx):
	    	self.lat = lat
    		self.lon = lon
    	self.box = box

    def show(self):
    	print('DataBox of shape: {shape}'.format(shape=self.box.shape)

    def add_buffer(self, p):
        from geopy.distance import vincenty

        edges = zip(self.lat[0, :], self.lon[0, :])
        edges.extend(zip(self.lat[:, -1], self.lon[:, -1]))
        edges.extend(zip(np.flipud(self.lat[-1, :]), np.flipud(self.lon[-1, :])))
        edges.extend(zip(np.flipud(self.lat[:, 0]), np.flipud(self.lon[:, 0])))

        for it in range(p.shape[0]):
            for ifeat in range(p.shape[1]):
                if np.isnan(p[it, ifeat, 'centroidY']):
                    continue
                center = p[it, ifeat, ['centroidY', 'centroidX']].values
                dist = min([vincenty(center, edge).kilometers for edge in edges])
                r = (p[it, ifeat, ['area']].values/np.pi)**.5
                if r>dist:
                    df0 = p[it,:,:]
                    for ichar in range(21):
                        df0.set_value(p.major_axis[ifeat], p.minor_axis[ichar], np.nan)
        return(p)

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

        if return_d:

        ll = np.array([X.flatten('F'), Y.flatten('F')]).T
        for i in range(box.shape[0]-1):
            hold = SpatialVx.make_SpatialVx(self.box[i,:,:], self.box[i+1,:,:], loc=ll)
            look = r_tools.FeatureFinder_gaussian(hold, nx=box.shape[2], ny=box.shape[1], 
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