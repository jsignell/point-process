import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Features:
    '''
    Parameters
    ----------
    p: pandas.Panel containing properties about all of the features and how they are matched
    databox: optional - a DataBox object with attributes: time, lat, lon, and box

    Return 
    '''
    def __init__(self, p, databox=None):
        self.p = p
        self.databox = databox

    def titanize(self):
        '''
        TITAN storm tracking output has a particular format with features sorted by complex 
        tracking numbers rather than by date. This function mimics that behavior by adding 
        a ComplexNum column to the feature Panel, reshaping it to a dataframe, and eliminating
        all the null tracking values. 
        '''
        df_by_time = self.p.to_frame(filter_observations=False).T

        n=0
        for it, t in enumerate(self.p.items[:-1]):
            if self.p.items[it+1]- t > pd.Timedelta(minutes=5):
                continue
            df0 = self.p.iloc[it,:,8].dropna()
            for nfeat0 in df0.index:
                if it==0:
                    df_by_time.set_value(t, (nfeat0, 'ComplexNum'), n)
                    n+=1
                df1 = self.p.iloc[it+1,:,:][self.p.iloc[it+1,:,7] == df0[nfeat0]]
                for nfeat1 in df1.index.values:
                    #print(nfeat0, nfeat1)
                    try:
                        df_by_time.loc[t, (nfeat0, 'ComplexNum')]
                    except:
                        df_by_time.set_value(t, (nfeat0, 'ComplexNum'), np.nan)
                    if ~ np.isnan(df_by_time.loc[t, (nfeat0, 'ComplexNum')]):
                        df_by_time.set_value(self.p.items[it+1], (nfeat1, 'ComplexNum'), df_by_time.loc[t, (nfeat0, 'ComplexNum')])
                    else:
                        #print 'incrementing n'
                        df_by_time.set_value(t, (nfeat0, 'ComplexNum'), n)
                        df_by_time.set_value(self.p.items[it+1], (nfeat1, 'ComplexNum'), n)
                        n+=1

        df_lightning = pd.concat([df_by_time[n] for n in self.p.major_axis]).dropna(how='all')
        df_light = df_lightning.reset_index().sort_values(['ComplexNum', 'index']).set_index([range(df_lightning.shape[0])])
        df_titanized = df_light[0:(df_light.ComplexNum.dropna().index[-1]+1)]
        return(df_titanized)

    def bearing_plot(self, ax=None, N=16, bottom=0):
        p = self.p
        if ax is None:
            ax = plt.subplot(111, polar=True)
        bearing = p[:,:,'Bearing']
        bearing[bearing<0] = bearing[bearing<0] + 360

        theta = np.linspace(0.0, 2 *np.pi, N, endpoint=False)
        radii, _ = np.histogram(bearing.unstack().dropna().values, bins=N)
        width = (2*np.pi) / N

        bars = ax.bar(theta, radii, width=width, bottom=bottom)
        ax.set_theta_zero_location("N")
        #ax.set_theta_direction(-1)

        # Use custom colors and opacity
        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.jet(r/float(np.max(radii))))
            bar.set_alpha(0.8)
        return(ax)
    
    def windrose(self, ax=None, N=16, bottom=0):
        if ax is None:
            ax = plt.subplot(111, polar=True)
        theta = np.linspace(0.0, 2 *np.pi, N, endpoint=False)
        ax.set_theta_zero_location("N")
        width = (2*np.pi) / N
        
        bearing = self.p[:,:,'Bearing'].stack()
        bearing[bearing<0] = bearing[bearing<0] + 360
        bearing.name = 'bearing'
        speed = self.p[:,:,'Centroid Distance'].stack()*100*12
        speed.name = 'speed'

        df = speed.to_frame().join(bearing)
        
        srange = zip([0,5,10,20,50], [5,10,20,50,100], ['#0000dd','green','#dddd00','#FF7800','#dd0000']) 
        ntot = df['bearing'].count()
        
        radii0 = [bottom]*N
        for smin, smax, c in srange:
            cond = ((df['speed']>=smin) & (df['speed']<smax))
            radii, _ = np.histogram(df['bearing'][cond].values, bins=N)
            radii = radii/float(ntot)*100
            bars = ax.bar(theta, radii, width=width, bottom=radii0, facecolor=c, alpha=0.8)
            #print smin, smax, c, radii
            radii0+= radii
        return(ax)

#     def get_storm_tracks(self, time=None):
#         p = self.p
#         tracks = []
#         for it in range(p.shape[0]-1):
#             try:
#                 df0 = p[self.databox.time[it],:,['centroidX', 'centroidY', 'Forecast']].dropna()
#                 df1 = p[self.databox.time[it+1],:,['centroidX', 'centroidY', 'Observed']].dropna()
#             except:
#                 continue
#             df0.index = df0['Forecast']
#             df1.index = df1['Observed']
#             df = df0.join(df1, lsuffix='_start', rsuffix='_end').dropna(how='any')
#             tracks.append(df)
#         return(tracks)

#     def plot_storm_tracks(self, ax, c='red', zorder=10):
#         p = self.p
#         tracks = self.get_storm_tracks()
#         for i in range(len(tracks)-1):
#             for ifeat in range(tracks[i].shape[0]):
#                 ax.plot(tracks[i].iloc[ifeat,[0,3]].values, tracks[i].iloc[ifeat,[1,4]].values, c=c, zorder=zorder)
#         return(ax)

    def get_dirs4(self):
        p = self.p
        ne = [-90, 0, 'North East']
        se = [-180, -90, 'South East']
        sw = [90, 180, 'South West']
        nw = [0, 90, 'North West']
        dirs4 = [nw, ne, sw, se]
        n = 1
        for b in dirs4:
            bool_array = ((p[:,:,'Bearing']>b[0]) & (p[:,:,'Bearing']<b[1]))
            b.append(bool_array)
            b.append(n)
            n+=1
        self.dirs4 = dirs4

    # def get_dirs8(self):
    #     p = self.p
    #     n = [-15, 15, 'North']
    #     ne = [30, 60, 'North East']
    #     e = [75, 105, 'East']
    #     se = [120, 150, 'South East']
    #     s = [165, -165, 'South']
    #     sw = [-150, -120, 'South West']
    #     w = [-105, -75, 'West']
    #     nw = [-60, -30, 'North West']
    #     dirs8 = [nw, n, ne, w, None, e, sw, s, se]
    #     n = 1
    #     for i, b in enumerate(dirs8):
    #         if i == 4:
    #             n+=1
    #             continue
    #         elif i == 7:
    #             bool_array = ((p[:,:,'Bearing']>b[0]) | (p[:,:,'Bearing']<b[1])) 
    #         else:
    #             bool_array = ((p[:,:,'Bearing']>b[0]) & (p[:,:,'Bearing']<b[1]))
    #         b.append(bool_array)
    #         b.append(n)
    #         n+=1
    #     self.dirs8 = dirs8

    def get_lon_lat(self, b, pos=False, neg=False, metrics=['area', 'Intensity0.9', 'Intensity0.25']):
        p = self.p
        lon=p[:,:,'centroidX'][b[3]].stack().values
        lat=p[:,:,'centroidY'][b[3]].stack().values
        if pos or neg:
            ifeats = np.where(b[3].values)[0]
            its = np.where(b[3].values)[1]
            j=[]
            dist=[]
            for metric in metrics:
                for ifeat, it in zip(ifeats, its):
                    try:
                        nfeat = p[it,ifeat,:].Forecast-1
                        j0 = p[it,ifeat,metric]
                        j1 = p[it+1,nfeat,metric]
                        if np.isnan(j1):
                            j.append(0)
                        else:
                            j.append(j1-j0)
                            dist.append(p[it,ifeat, ['Mean Error Distance']].values)
                    except:
                        j.append(0)
            j = np.array(j)
        else:
            return(lon, lat, np.nanmean(p[:,:, ['Mean Error Distance']].values))
        pos_lon = np.concatenate([lon]*len(metrics))[j>0]
        pos_lat = np.concatenate([lat]*len(metrics))[j>0]
        neg_lon = np.concatenate([lon]*len(metrics))[j<0]
        neg_lat = np.concatenate([lat]*len(metrics))[j<0]
        if pos and not neg:
            return(pos_lon, pos_lat, np.mean(dist))
        elif neg and not pos:
            return(neg_lon, neg_lat, np.mean(dist))
        elif neg and pos:
            return(pos_lon, pos_lat, neg_lon, neg_lat, np.mean(dist))

    def plot_kde(self, pos, neg, lon=None, lat=None, metrics=['area', 'Intensity0.9', 'Intensity0.25'], **kwargs):
        import cartopy.crs as ccrs
        from plotting import choose_cmap, background, urban
        fig = plt.figure(figsize=(12,8))
        axes = []
        for b in self.dirs4:
            ax = plt.subplot(2, 2, b[4], projection=ccrs.PlateCarree())
            if pos and neg:
                pos_lon, pos_lat, neg_lon, neg_lat, dist = self.get_lon_lat(b, pos, neg, metrics)
                if self.databox:
                    xx, yy, pos_f = self.databox.kde(pos_lon, pos_lat)    
                    xx, yy, neg_f = self.databox.kde(neg_lon, neg_lat) 
                else:
                    xx, yy, pos_f = kde(lon, lat, pos_lon, pos_lat)    
                    xx, yy, neg_f = kde(lon, lat, neg_lon, neg_lat) 
                f = (pos_f-neg_f)
                flim = np.ceil(max(np.abs(f.min()), f.max()))
                nfeats = pos_lon.shape[0]+neg_lon.shape[0]
            else:
                _lon, _lat, dist = self.get_lon_lat(b, pos, neg, metrics)
                if self.databox:
                    xx, yy, f = self.databox.kde(_lon, _lat)
                else:
                    xx, yy, f = kde(lon, lat, _lon, _lat)
                nfeats = _lon.shape[0]
            cfset = ax.contourf(xx, yy, f, cmap=choose_cmap(pos, neg), zorder=3, **kwargs)
            background(ax)
            urban(ax, facecolor='None', linewidth=2, edgecolor='red')
            ax.set_title('From the {direction}: {k} features found'.format(direction=b[2], k=nfeats))
            CB = plt.colorbar(cfset, ax=ax) 
            axes.append(ax)
        return(fig)

def kde(lon, lat, _lon, _lat):
    import scipy.stats as st
    xx, yy = lon, lat
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([_lon, _lat])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    return(xx,yy,f)    
