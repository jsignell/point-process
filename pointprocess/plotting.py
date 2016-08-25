import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.img_tiles import GoogleTiles, StamenTerrain

def background_scale(ax, scale='110m'):
    '''
    Add standard background features to geoAxes object
    '''
    states = cfeature.NaturalEarthFeature(category='cultural',
                                          name='admin_1_states_provinces_lines',
                                          scale=scale,
                                          facecolor='none')
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale,
                                         edgecolor='None', facecolor=cfeature.COLORS['water'])
    coast = cfeature.NaturalEarthFeature('physical', 'coastline', scale,
                                         edgecolor='k', facecolor='None')
    ax.add_feature(ocean, zorder=4)
    ax.add_feature(coast, zorder=6)
    ax.add_feature(states, zorder=7)
    gl = ax.gridlines(draw_labels=True, zorder=3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    return(ax)

def background(ax):
    '''
    Add standard background features to geoAxes object
    '''
    states = cfeature.NaturalEarthFeature(category='cultural',
                                          name='admin_1_states_provinces_lines',
                                          scale='10m',
                                          facecolor='none')
    ax.add_feature(cfeature.OCEAN, zorder=4)
    ax.add_feature(cfeature.COASTLINE, zorder=6)

    ax.add_feature(cfeature.BORDERS, zorder=7)
    ax.add_feature(states, zorder=8)
    gl = ax.gridlines(draw_labels=True, zorder=3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    return(ax)

def pre_shaded(ax, fname, zorder=2):
    img_extent = ([-180, 180, -90, 90])
    img = plt.imread(fname)
    ax.imshow(img, origin='upper', extent=img_extent,
              transform=ccrs.PlateCarree(), cmap='Greys_r', zorder=zorder)
    return(ax)

class ShadedReliefESRI(GoogleTiles):
    """
    I struggled for a WHILE to find a basemap tiling service that I was happy
    with. There are some good options available through Mapbox, but the best
    plain, no label, shaded relief that I found is provided by ESRI. Check out
    their license before you use this tiler.
    """
    # shaded relief
    def _image_url(self, tile):
        x, y, z = tile
        url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
               'World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}').format(
               z=z, y=y, x=x)
        return url

def urban(ax, **kwargs):
    urban = cfeature.NaturalEarthFeature(category='cultural',
                                         name='urban_areas',
                                         scale='10m',
                                         edgecolor='red',
                                         facecolor='None')
    ax.add_feature(urban, zorder=10, **kwargs)
    return(ax)

def choose_cmap(pos, neg):
    if pos and not neg:
        cmap="PuRd"
    elif neg and not pos:
        cmap="PuBu"
    elif not neg and not pos:
        cmap="Greys"
    elif neg and pos:
        cmap="RdBu_r"
    return(cmap)

def plot_contour(lat, lon, grid, ax=None, extent=None,
                tiler=StamenTerrain(), zoom=6,
                N=7, fontsize=10, fmt='%1.1f', **kwargs):
    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.axes(projection=ccrs.PlateCarree())
    if extent is not None:
        xmin, xmax, ymin, ymax = extents
        x = lon[xmin:xmax]
        y = lat[ymin:ymax]
        z = grid[ymin:ymax, xmin:xmax]
    elif lon.shape[0] != grid.shape[1]:
        x = (lon[:-1]+lon[1:])/2.
        y = (lat[:-1]+lat[1:])/2.
        z = grid
    else:
        x = lon
        y = lat
        z = grid
    CS = plt.contour(x, y, z, N, cmap='Greys', **kwargs)
    plt.clabel(CS, inline=1, fontsize=fontsize, fmt=fmt)
    ax.add_image(tiler, zoom)
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False
    return ax

def plot_grid(lat, lon, grid, ax=None, cbar=False, interpolation='None', zorder=5, **kwargs):
    '''
    Simple and fast plot generation for gridded data **MUST BE RECTANGULAR**

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
    if ax is None:
        ax = background(plt.axes(projection=ccrs.PlateCarree()))
    if (np.array(lat).ndim == 1) or (lat[0,0] == lat[-1,0]):
        im = ax.imshow(grid, interpolation=interpolation,
                       extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                       zorder=zorder,
                       **kwargs)
    else:
        im = ax.pcolor(lon, lat, grid, zorder=zorder, **kwargs)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
    if cbar:
        plt.colorbar(im, ax=ax)
    return im, ax

def windrose_cbar(fig=None):
    '''
    If you have a figsize in mind, then pass a figure object to this function
    and the colorbar will be drawn to suit
    '''
    import matplotlib.patches as mpatch

    if fig is None:
        fig = plt.figure(figsize=(8,1))
    y = fig.get_figwidth()

    srange = zip([0,5,10,20,50], [5,10,20,50,100], ['#0000dd','green','#dddd00','#FF7800','#dd0000'])
    n=1
    for smin, smax, c in srange:
        ax = plt.subplot(1,5,n)
        patch = mpatch.FancyBboxPatch([0,0], 1, 1, boxstyle='square', facecolor=c)
        ax.add_patch(patch)
        plt.axis('off')
        if y>=12:
            ax.text(.1, .4, '{smin} - {smax} km/hr'.format(smin=smin, smax=smax),
                    fontsize=min(18, y+2), fontdict = {'color': 'white'})
        else:
            ax.text(.1, .4, '{smin} - {smax}\nkm/hr'.format(smin=smin, smax=smax),
                    fontsize=min(14, y+5), fontdict = {'color': 'white'})
        n+=1

def feature_locations(df, ax=None, figsize=(14,8), tiler=StamenTerrain(),
                      lat='centroidY' ,lon='centroidX', paths=False,
                      features=True, zoom=6, zorder=5, colorby='ComplexNum',
                      c='k'):
    '''
    Use a computed titanized dataframe to show all the features and their paths
    '''
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        background(ax)
        ax.add_image(tiler, zoom)
    if features:
        storm_names = dict([(n[1], n[0]) for n in enumerate(df[colorby].unique())])
        df.plot.scatter(x=lon, y=lat,
                        c=[storm_names[n] for n in df[colorby]],
                        ax=ax, cmap='rainbow',
                        edgecolor='None', s=50, zorder=zorder)
    if paths:
        gb = df.groupby(df['ComplexNum'])
        for k,v in gb.groups.items():
            gb.get_group(k).plot(x=lon, y=lat, c=c, ax=ax, legend=None, zorder=zorder+1)
    return(ax)
