import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import geopandas as gpd
import cartopy as crtpy
import sys, os
from pyproj import Transformer
from matplotlib.widgets import Button

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
# which_norm = mcolors.BoundaryNorm(boundaries=np.arange(5)-0.5, ncolors=4)

# ================================== Define file paths and load data ==================================

save = False
COMP_path_arr = []
if sys.argv[1] in ["s", "m"]:
  if sys.argv[1] == "s":
    COMP_path_arr.append(sys.argv[2])
  
  if sys.argv[1] == "m":
    for root, dirs, files in os.walk(sys.argv[2]):
      for file in files:
         if file.endswith(".nc"):
           COMP_path_arr.append(os.path.join(root, file))
  
  save = True
  SAVE_dir = sys.argv[3]

else:
  COMP_path_arr.append(sys.argv[1])

COMP_path_arr.sort()

# load static resources once outside the loop
# shapefile data and projection are constant for all composites
shape_file = "/home/nvm/nvm_local/data/comarques_shape/2025/divisions-administratives-v2r1-comarques-50000-20250730.shp"
comarques = gpd.read_file(shape_file)
comarques = comarques.to_crs(epsg=25831)

# transformer and radar positions are also fixed
_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25831", always_xy=True)
rad_pos = np.array([[41.60192013,  1.40283002],
                    [41.37334999,  1.88197011],
                    [41.8888201 ,  2.99717009],
                    [41.09175006,  0.86348003]])
rad_x, rad_y = _to_utm.transform(rad_pos[:,1], rad_pos[:,0])

for COMP_path in COMP_path_arr:
  filename = os.path.basename(COMP_path)[:-3]

  # ============================== Load composite data and radar positions ==============================

  # open dataset and immediately extract all arrays needed; close file promptly
  with xr.open_dataset(COMP_path, engine="scipy") as ds:
      Z_comp = ds.Z.values
      QI_comp = ds.QI.values
      which_rad = ds.RAD.values
      ELEV = ds.ELEV.values
      x_vals = ds.x.values
      y_vals = ds.y.values

  # free the dataset object and let GC reclaim it
  del ds

  # =========================================== Create plots ===========================================

  # Create figure and subplots with custom layout
  fig = plt.figure(figsize=(20,10))

  # State variable for colormap bounds toggle
  colormap_state = {'use_zero_start': False}
  gs = gridspec.GridSpec(
      2, 3, figure=fig,
      width_ratios=[5, 2.5, 2.5],   # big left + two small columns
      height_ratios=[2, 2],
      left=0.03, right=0.97, wspace=0.1, hspace=-0.1,
      top=0.92, bottom=0.05
  )

  proj = crtpy.crs.UTM(zone=31)

  # Big plot (left, spans both rows)
  ax_big = fig.add_subplot(gs[:, 0], projection=proj)

  # Small 2x2 on the right
  ax_tl = fig.add_subplot(gs[0, 1], projection=proj)  # top-left small
  ax_tr = fig.add_subplot(gs[0, 2], projection=proj)  # top-right small (empty)
  ax_bl = fig.add_subplot(gs[1, 1], projection=proj)  # bottom-left small
  ax_br = fig.add_subplot(gs[1, 2], projection=proj)  # bottom-right small

  axes_all = [ax_big, ax_tl, ax_bl, ax_br, ax_tr]

  shape_file = "/home/nvm/nvm_local/data/comarques_shape/2025/divisions-administratives-v2r1-comarques-50000-20250730.shp"
  comarques = gpd.read_file(shape_file)
  comarques = comarques.to_crs(epsg=25831)

  for axis in axes_all:
      comarques.boundary.plot(ax=axis, color='black', linewidth=0.7)
      axis.coastlines(linewidth=1.5)
      axis.add_feature(crtpy.feature.BORDERS, linestyle='-', edgecolor='black', linewidth=1.5)
      axis.set_xticks([]), axis.set_yticks([])
      axis.set_xlim(x_vals.min(), x_vals.max())
      axis.set_ylim(y_vals.min(), y_vals.max())

  # --- PLOTS (re-mapped from old positions) ---

  # BIG: old ax[0,0] -> Z composite
  Z_comp_plot = np.copy(Z_comp)
  Z_comp_plot[Z_comp == -32] = np.nan
  pc = ax_big.pcolormesh(x_vals, y_vals, Z_comp_plot, norm=rad_norm, cmap=rad_cmap)
  cbar_big = fig.colorbar(pc, ax=ax_big, fraction=0.035, pad=0.01, boundaries=bounds, ticks=bounds)
  ax_big.set_title(filename)
  pc_handle = pc  # Store for later update
  Z_comp_original = np.copy(Z_comp)  # Store original data for toggling

  # SMALL top-left: old ax[0,1] -> which radar
  pc = ax_tl.pcolormesh(x_vals, y_vals, which_rad, cmap=which_cmap)
  cbar = fig.colorbar(pc, ax=ax_tl, ticks=np.arange(0.75/2, 3, 0.75), fraction=0.03, pad=0.01)
  cbar.ax.set_yticklabels(labels)
  ax_tl.set_title("Which radar data is selected")

  # SMALL bottom-left: old ax[1,0] -> QI DET
  QI_DET, QI_UNDET = np.copy(QI_comp), np.copy(QI_comp)
  QI_DET[Z_comp == -32] = np.nan
  QI_UNDET[Z_comp > -32] = np.nan

  pc = ax_bl.pcolormesh(x_vals, y_vals, QI_DET, vmin=0, vmax=1, cmap=cmap_QI)
  fig.colorbar(pc, ax=ax_bl, fraction=0.03, pad=0.01)
  ax_bl.set_title("QI DETECTED composite")

  # SMALL bottom-right: old ax[1,1] -> QI UNDET
  pc = ax_br.pcolormesh(x_vals, y_vals, QI_UNDET, vmin=0, vmax=1, cmap=cmap_QI)
  fig.colorbar(pc, ax=ax_br, fraction=0.03, pad=0.01)
  ax_br.set_title("QI UNDETECTED composite")

  # EMPTY top-right subplot
  pc = ax_tr.pcolormesh(x_vals, y_vals, ELEV, vmin=0.6, vmax=3, cmap=cmap_elev)
  tick_positions = np.arange(len(values)) + 0.5
  cb = fig.colorbar(
      plt.cm.ScalarMappable(norm=norm_elev, cmap=cmap_elev),
      ax=ax_tr,
      ticks=tick_positions,
      fraction=0.03, pad=0.01
  )
  cb.set_ticklabels([str(v) for v in values])
  ax_tr.set_title("Elevation (deg)")

  # Radar positions on all *real* axes
  for axis in [ax_big, ax_tl, ax_bl, ax_br, ax_tr]:
      axis.scatter(rad_x, rad_y, facecolors="white",
                  edgecolors="black", linewidths=2, zorder=20)

  if save:
    plt.savefig(f"{SAVE_dir}/{filename}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)  # explicitly close this figure
    # clean up large arrays and force garbage collection
    del Z_comp, QI_comp, which_rad, ELEV, x_vals, y_vals
    import gc; gc.collect()
    print(f"{filename}.png")
  else:
    # Add toggle button for colormap bounds only when not saving (inline plotting)
    def toggle_colormap_bounds(event):
      """Toggle between full bounds (-10 to 65) and zero-start bounds (5 to 65)
      Also toggles NaN handling: -32 dBZ to NaN vs values < 5 dBZ to NaN"""
      colormap_state['use_zero_start'] = not colormap_state['use_zero_start']
      
      if colormap_state['use_zero_start']:
        # Show only from 5 dBZ upwards, set values < 5 dBZ to NaN
        Z_comp_plot_new = np.copy(Z_comp_original)
        Z_comp_plot_new[Z_comp_original < 5] = np.nan
        pc_handle.set_array(Z_comp_plot_new.ravel())
        pc_handle.set_cmap(rad_cmap_zero)
        pc_handle.set_norm(rad_norm_zero)
        pc_handle.set_clim(vmin=5, vmax=65)
        button_toggle.label.set_text('-10\ndBZ')
      else:
        # Show from -10 dBZ (all echoes), set -32 to NaN
        Z_comp_plot_new = np.copy(Z_comp_original)
        Z_comp_plot_new[Z_comp_original == -32] = np.nan
        pc_handle.set_array(Z_comp_plot_new.ravel())
        pc_handle.set_cmap(rad_cmap)
        pc_handle.set_norm(rad_norm)
        pc_handle.set_clim(vmin=-10, vmax=65)
        button_toggle.label.set_text('5\ndBZ')
      
      fig.canvas.draw_idle()
    
    # Create button
    ax_button = fig.add_axes([0.455, 0.16, 0.016, 0.03]) # left, bottom, width, height
    button_toggle = Button(ax_button, '5\ndBZ')
    button_toggle.label.set_fontsize(10)
    button_toggle.on_clicked(toggle_colormap_bounds)
    
    plt.show()