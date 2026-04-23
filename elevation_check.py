import rasterio
import numpy as np
with rasterio.open('outputs/dem_reprojected.tif') as src:
    data = src.read(1).astype(float)
    nodata = src.nodata or -9999
    valid = data[data != nodata]
    print(f'Min elevation: {valid.min():.1f} m')
    print(f'Max elevation: {valid.max():.1f} m')
    print(f'Mean elevation: {valid.mean():.1f} m')
    print(f'25th percentile: {np.percentile(valid, 25):.1f} m')
    print(f'75th percentile: {np.percentile(valid, 75):.1f} m')
