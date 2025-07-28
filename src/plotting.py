import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


markercolor_list = [
    "#1f77b4",  # Blue
    "#d62728",  # Red
    "#2ca02c",  # Green
    "#ff7f0e",  # Orange
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#17becf",  # Cyan
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
]

background_colors = [
    "#e6f0fa",  # Light Blue
    "#fdecea",  # Light Red
    "#eaf6ea",  # Light Green
    "#fff3e0",  # Light Orange
    "#f3eaf6",  # Lavender
    "#f2ebe7",  # Light Brown
    "#fdebf3",  # Light Pink
    "#e6f9fb",  # Light Cyan
    "#f2f2f2",  # Light Gray
    "#f9f9e3",  # Light Olive
]


def plot_tx_signal(config, tx_signal):
  """ Plotting function for the tx signal with mutiple sections
  """
  # tx signal indexing
  idx_stf_begin = int(config.span/2)
  idx_stf_end   = idx_stf_begin + config.reps_STF * config.N_STF
  idx_ltf_end   = idx_stf_end + config.reps_LTF*config.N_LTF
  idx_pilot_end = idx_ltf_end + config.N_pilots
  idx_syms_end  = idx_pilot_end + config.N_syms
  idx_zeros_end = idx_syms_end + int(config.N_syms/10)

  print(idx_ltf_end*config.sps)
  t = np.arange(0, len(tx_signal))
  fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6), constrained_layout=True)
  fig.suptitle("rrc pulse shaped transmitted signal")

  # STF stem
  markerline1, stemlines1, baseline1 = axs[0].stem(
      np.arange(start=idx_stf_begin*config.sps, step=config.sps, stop=idx_stf_end*config.sps),
      tx_signal.real[::config.sps][idx_stf_begin:idx_stf_end],
      markerfmt='.', linefmt='', label='Short Training Field'
  )
  plt.setp(markerline1, markersize=3)
  markerline1.set_color(markercolor_list[0])
  stemlines1.set_visible(False)
  baseline1.set_visible(False)

  # LTF stem
  markerline2, stemlines2, baseline2 = axs[0].stem(
      np.arange(start=idx_stf_end*config.sps, step=config.sps, stop=idx_ltf_end*config.sps),
      tx_signal.real[::config.sps][idx_stf_end:idx_ltf_end],
      markerfmt='.', linefmt='', label='Long Training Field'
  )
  plt.setp(markerline2, markersize=3)
  markerline2.set_color(markercolor_list[1])
  stemlines2.set_visible(False)
  baseline2.set_visible(False)

  # Pilot stem
  markerline3, stemlines3, baseline3 = axs[0].stem(
      np.arange(start=idx_ltf_end*config.sps, step=config.sps, stop=idx_pilot_end*config.sps),
      tx_signal.real[::config.sps][idx_ltf_end:idx_pilot_end],
      markerfmt='.', linefmt='', label='Pilots'
  )
  plt.setp(markerline3, markersize=3)
  markerline3.set_color(markercolor_list[2])
  stemlines3.set_visible(False)
  baseline3.set_visible(False)

  # Data stem
  markerline_syms_real, stemlines_syms_real, baseline_syms_real = axs[0].stem(
      np.arange(start=idx_pilot_end*config.sps, step=config.sps, stop=idx_syms_end*config.sps),
      tx_signal.real[::config.sps][idx_pilot_end:idx_syms_end],
      markerfmt='.', linefmt='', label='Data Symbols'
  )
  plt.setp(markerline_syms_real, markersize=3)
  markerline_syms_real.set_color(markercolor_list[3])
  stemlines_syms_real.set_visible(False)
  baseline_syms_real.set_visible(False)

  # verical lines for separating stf and ltf
  idx_stf_separating_dashed = t[idx_stf_begin*config.sps:idx_stf_end*config.sps:int((idx_stf_end-idx_stf_begin)*config.sps/config.reps_STF)]
  idx_ltf_separating_dashed = t[idx_stf_end*config.sps:idx_ltf_end*config.sps:int((idx_ltf_end-idx_stf_end)*config.sps/config.reps_LTF)]

  # Formatting
  axs[0].axhline(0, color='black', linewidth=0.5)
  axs[0].axvline(0, color='black', linewidth=0.5)
  for x in idx_stf_separating_dashed:
    axs[0].axvline(x, color='gray', linestyle='--', linewidth=1)
  for x in idx_ltf_separating_dashed:
    axs[0].axvline(x, color='gray', linestyle='--', linewidth=1)
  axs[0].axvspan(idx_stf_begin*config.sps, idx_stf_end*config.sps, facecolor=background_colors[0], alpha=1)
  axs[0].axvspan(idx_stf_end*config.sps, idx_ltf_end*config.sps, facecolor=background_colors[1], alpha=1)
  axs[0].axvspan(idx_ltf_end*config.sps, idx_pilot_end*config.sps, facecolor=background_colors[2], alpha=1)
  axs[0].axvspan(idx_pilot_end*config.sps, idx_syms_end*config.sps, facecolor=background_colors[3], alpha=1)
  axs[0].set_title("Tx signal pulse train: real")
  axs[0].set_ylabel("Real")
  axs[0].grid(True)
  axs[0].set_ylim([-3, 3])

  # STF (Imag part)
  markerline4, stemlines4, baseline4 = axs[1].stem(
      np.arange(start=0, step=config.sps, stop=(int(config.span/2)+config.reps_STF*config.N_STF)*config.sps),
      tx_signal.imag[::config.sps][:(int(config.span/2)+config.reps_STF*config.N_STF)],
      markerfmt='r.', linefmt='', label='Short Training Field'
  )
  plt.setp(markerline4, markersize=3)
  markerline4.set_color(markercolor_list[0])
  stemlines4.set_visible(False)
  baseline4.set_visible(False)

  # LTF (Imag part)
  markerline5, stemlines5, baseline5 = axs[1].stem(
      np.arange(start=(int(config.span/2)+config.reps_STF*config.N_STF)*config.sps, step=config.sps,
                stop=(int(config.span/2)+config.reps_STF*config.N_STF+config.reps_LTF*config.N_LTF)*config.sps),
      tx_signal.imag[::config.sps][(int(config.span/2)+config.reps_STF*config.N_STF):(int(config.span/2)+config.reps_STF*config.N_STF+config.reps_LTF*config.N_LTF)],
      markerfmt='b.', linefmt='', label='Long Training Field'
  )
  plt.setp(markerline5, markersize=3)
  markerline5.set_color(markercolor_list[1])
  stemlines5.set_visible(False)
  baseline5.set_visible(False)

  # Pilot stem (Imag part)
  markerline6, stemlines6, baseline6 = axs[1].stem(
      np.arange(start=idx_ltf_end*config.sps, step=config.sps, stop=idx_pilot_end*config.sps),
      tx_signal.imag[::config.sps][idx_ltf_end:idx_pilot_end],
      markerfmt='.', linefmt='', label='Pilots'
  )
  plt.setp(markerline6, markersize=3)
  markerline6.set_color(markercolor_list[2])
  stemlines6.set_visible(False)
  baseline6.set_visible(False)

  # Data stem (Imag part)
  markerline_syms_imag, stemlines_syms_imag, baseline_syms_imag = axs[1].stem(
      np.arange(start=idx_pilot_end*config.sps, step=config.sps, stop=idx_syms_end*config.sps),
      tx_signal.imag[::config.sps][idx_pilot_end:idx_syms_end],
      markerfmt='.', linefmt='', label='Data Symbols'
  )
  plt.setp(markerline_syms_imag, markersize=3)
  markerline_syms_imag.set_color(markercolor_list[3])
  stemlines_syms_imag.set_visible(False)
  baseline_syms_imag.set_visible(False)

  # Formatting
  axs[1].axhline(0, color='black', linewidth=0.5)
  axs[1].axvline(0, color='black', linewidth=0.5)
  for x in idx_stf_separating_dashed:
    axs[1].axvline(x, color='gray', linestyle='--', linewidth=1)
  for x in idx_ltf_separating_dashed:
    axs[1].axvline(x, color='gray', linestyle='--', linewidth=1)
  axs[1].set_title("Tx signal pulse train: imag")
  axs[1].set_ylabel("Imaginary")
  axs[1].grid(True)
  axs[1].set_ylim([-3, 3])

  # Region highlighting (same as above, optional)
  axs[1].axvspan(idx_stf_begin*config.sps, idx_stf_end*config.sps, facecolor=background_colors[0], alpha=1)
  axs[1].axvspan(idx_stf_end*config.sps, idx_ltf_end*config.sps, facecolor=background_colors[1], alpha=1)
  axs[1].axvspan(idx_ltf_end*config.sps, idx_pilot_end*config.sps, facecolor=background_colors[2], alpha=1)
  axs[1].axvspan(idx_pilot_end*config.sps, idx_syms_end*config.sps, facecolor=background_colors[3], alpha=1)

  # Region patches (shared for both)
  region_patches = [
      Patch(facecolor=background_colors[0], edgecolor='black', label='STF region', alpha=1),
      Patch(facecolor=background_colors[1], edgecolor='black', label='LTF region', alpha=1),
      Patch(facecolor=background_colors[2], edgecolor='black', label='Pilot region', alpha=1),
      Patch(facecolor=background_colors[3], edgecolor='black', label='Data region', alpha=1),
  ]

  marker_handles = [
    Line2D([0], [0], marker='o', color='w', label='Short Training Field', markerfacecolor=markercolor_list[0], markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Long Training Field', markerfacecolor=markercolor_list[1], markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Pilot Symbols', markerfacecolor=markercolor_list[2], markersize=5),
    Line2D([0], [0], marker='o', color='w', label='Data Symbols', markerfacecolor=markercolor_list[3], markersize=5),
  ]
  # Combine all handles and labels
  all_handles = region_patches + marker_handles

  # Create shared legend outside the plot
  axs[1].legend(handles=all_handles,
           loc='lower center',
           bbox_to_anchor=(0.5, 0),
           ncol=3,
           frameon=True)

  # Adjust layout to make room for legend
  # fig.tight_layout(rect=[0, 0.1, 1, 1])

  fig.supxlabel("Symbol Index")
  # plt.savefig("figs/rrc_pulse_shaped_transmitted_signal.png")
  plt.show()