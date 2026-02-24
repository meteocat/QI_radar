import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import cartopy as crtpy
import os
from pyproj import Transformer

# ================================ Define colormaps and values for plotting ================================
bounds = [-10, -5, 0, 5, 10, 15, 20, 25, 30,
          35, 40, 45, 50, 55, 60, 65]
colors = [
    "#646464",  # -10–0 dBZ (gray, very weak)
    "#04e9e7",  # 0
    "#019ff4",  # 5
    "#0300f4",  # 10
    "#02fd02",  # 15
    "#01c501",  # 20
    "#008e00",  # 25
    "#fdf802",  # 30
    "#e5bc00",  # 35
    "#fd9500",  # 40
    "#fd0000",  # 45
    "#d40000ff",  # 50
    "#bc0000",  # 55
    "#f800fd",  # 60
    "#9854c6",  # 65
]
rad_cmap = mcolors.ListedColormap(colors)
rad_norm = mcolors.BoundaryNorm(bounds, rad_cmap.N)

# Alternative bounds starting from 0 dBZ (skipping first color and bounds)
bounds_zero = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
colors_zero = colors[3:]  # Skip the gray color for -10 to 0 dBZ
rad_cmap_zero = mcolors.ListedColormap(colors_zero)
rad_norm_zero = mcolors.BoundaryNorm(bounds_zero, rad_cmap_zero.N)

def make_discrete_cmap(cmap, N):
    """
    Converteix un colormap continu en un colormap discret amb N colors.
    Arguments:
      cmap : nom de colormap (str) o objecte matplotlib.colors.Colormap
      N    : nombre d'esglaons (int)
    Retorna:
      matplotlib.colors.ListedColormap
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # mostrem els N colors muestrejant uniformement el colormap
    colors = cmap(np.linspace(0, 1, N))
    return mcolors.ListedColormap(colors, name=f"{cmap.name}_discrete_{N}")

cmap_QI = make_discrete_cmap('RdYlGn', 10)

values = [0.6, 0.8, 1.0, 1.3, 1.7, 2.0, 3.0]
colors = ['#3288bd','#66c2a5','#abdda4','#e6f598',
          '#fee08b','#f46d43','#d53e4f']
fake_bounds = np.arange(len(values)+1)
cmap_elev = mcolors.ListedColormap(colors)
norm_elev = mcolors.BoundaryNorm(fake_bounds, cmap_elev.N)

colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]  # Blue, Green, Orange, Red
labels = ['CDV', 'PBE', 'PDA', 'LMI']
which_cmap = mcolors.ListedColormap(colors)

# ================================== Plotting function ==================================

def plot_composite_file(nc_path: str, save_dir: str, comarques, large_cmap: bool = True):
    """Generate and save the composite figure from a netCDF file.

    Parameters
    ----------
    nc_path : str
        Path to ``.nc`` file containing the composite fields.
    save_dir : str
        Directory where the PNG will be stored.  Created if missing.
    large_cmap : bool, optional
        If ``True`` use the full -10..65 dBZ colour- scale.  Otherwise drop
        the first three colours and mask values below 5 dBZ (reduced scale).
    """

    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(nc_path)[:-3]

    # read data arrays
    with xr.open_dataset(nc_path, engine="scipy") as ds:
        Z_comp = ds.Z.values
        QI_comp = ds.QI.values
        which_rad = ds.RAD.values
        ELEV = ds.ELEV.values
        x_vals = ds.x.values
        y_vals = ds.y.values

    # geographic overlays and radar positions
    _to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
    rad_pos = np.array([[41.60192013,  1.40283002],
                        [41.37334999,  1.88197011],
                        [41.8888201 ,  2.99717009],
                        [41.09175006,  0.86348003]])
    rad_x, rad_y = _to_utm.transform(rad_pos[:,1], rad_pos[:,0])

    if large_cmap:
        cmap = rad_cmap
        norm = rad_norm
        mask = (Z_comp == -32)
        bounds_used = bounds
    else:
        cmap = rad_cmap_zero
        norm = rad_norm_zero
        mask = (Z_comp < 5)
        bounds_used = bounds_zero

    Z_comp_plot = np.copy(Z_comp)
    Z_comp_plot[mask] = np.nan

    # figure layout
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                            width_ratios=[5,2.5,2.5],
                            height_ratios=[2,2],
                            left=0.03, right=0.97, wspace=0.1, hspace=-0.1,
                            top=0.92, bottom=0.05)
    proj = crtpy.crs.UTM(zone=31)
    ax_big = fig.add_subplot(gs[:,0], projection=proj)
    ax_tl = fig.add_subplot(gs[0,1], projection=proj)
    ax_tr = fig.add_subplot(gs[0,2], projection=proj)
    ax_bl = fig.add_subplot(gs[1,1], projection=proj)
    ax_br = fig.add_subplot(gs[1,2], projection=proj)
    axes_all = [ax_big, ax_tl, ax_bl, ax_br, ax_tr]

    for axis in axes_all:
        comarques.boundary.plot(ax=axis, color='black', linewidth=0.7)
        axis.coastlines(resolution='10m')
        axis.add_feature(crtpy.feature.BORDERS, linestyle='-', edgecolor='black', linewidth=1.5)
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_xlim(x_vals.min(), x_vals.max())
        axis.set_ylim(y_vals.min(), y_vals.max())

    pc_big = ax_big.pcolormesh(x_vals, y_vals, Z_comp_plot, norm=norm, cmap=cmap)
    fig.colorbar(pc_big, ax=ax_big, fraction=0.03, pad=0.01,
                 boundaries=bounds_used, ticks=bounds_used)
    ax_big.set_title("REFLECTIVITY (dBZ)", fontsize=14)

    pc = ax_tl.pcolormesh(x_vals, y_vals, which_rad, cmap=which_cmap)
    cbar = fig.colorbar(pc, ax=ax_tl,
                        ticks=np.arange(0.75/2, 3, 0.75),
                        fraction=0.03, pad=0.01)
    cbar.ax.set_yticklabels(labels)
    ax_tl.set_title("RADAR SELECTED")

    QI_DET = np.copy(QI_comp)
    QI_UNDET = np.copy(QI_comp)
    QI_DET[Z_comp == -32] = np.nan
    QI_UNDET[Z_comp > -32] = np.nan

    pc = ax_bl.pcolormesh(x_vals, y_vals, QI_DET, vmin=0, vmax=1, cmap=cmap_QI)
    fig.colorbar(pc, ax=ax_bl, fraction=0.03, pad=0.01)
    ax_bl.set_title("QUALITY DETECTED")

    pc = ax_br.pcolormesh(x_vals, y_vals, QI_UNDET, vmin=0, vmax=1, cmap=cmap_QI)
    fig.colorbar(pc, ax=ax_br, fraction=0.03, pad=0.01)
    ax_br.set_title("QUALITY UNDETECTED")

    pc = ax_tr.pcolormesh(x_vals, y_vals, ELEV, vmin=0.6, vmax=3, cmap=cmap_elev)
    tick_positions = np.arange(len(values)) + 0.5
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_elev, cmap=cmap_elev),
                      ax=ax_tr, ticks=tick_positions,
                      fraction=0.03, pad=0.01)
    cb.set_ticklabels([str(v) for v in values])
    ax_tr.set_title("ELEVATION (deg)")

    for axis in [ax_big, ax_tl, ax_bl, ax_br, ax_tr]:
        axis.scatter(rad_x, rad_y, facecolors="white",
                     edgecolors="black", linewidths=2, zorder=20)

    outpath = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plt.clf()
    plt.close("all")
    del Z_comp, QI_comp, which_rad, ELEV, x_vals, y_vals
    import gc; gc.collect()
    # print(f"Saved {outpath}")


