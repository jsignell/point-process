library('SpatialVx')
library('stringr')

FeatureFinder_gaussian <- function(hold, nx, ny, ...) {
    look <- FeatureFinder(hold, smoothfun="gauss2dsmooth", smoothfunargs=c(nx=nx, ny=ny), ...)
    return(look)
}
