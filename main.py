import numpy as np
import matplotlib.pyplot as plt
import src.mapping as mapping
import src.synchronization as sync
from remoteRF.drivers.adalm_pluto import *

def main(tx_gain_dB=-25, M=64):

  # ---------------------------------------------------------------
  # Digital communication system parameters.
  # ---------------------------------------------------------------
  fs = 1e6     # baseband sampling rate (samples per second)
  ts = 1 / fs  # baseband sampling period (seconds per sample)
  sps = 10     # samples per data symbol
  T = ts * sps # time between data symbols (seconds per symbol)
  CFO = 80

  # ---------------------------------------------------------------
  # Pluto system parameters.
  # ---------------------------------------------------------------
  sample_rate = fs                # sampling rate, between ~600e3 and 61e6
  tx_carrier_freq_Hz = 915e6      # transmit carrier frequency, between 325 MHz to 3.8 GHz
  rx_carrier_freq_Hz = 915e6+CFO      # receive carrier frequency, between 325 MHz to 3.8 GHz
  tx_rf_bw_Hz = sample_rate * 1   # transmitter's RF bandwidth, between 200 kHz and 56 MHz
  rx_rf_bw_Hz = sample_rate * 1   # receiver's RF bandwidth, between 200 kHz and 56 MHz
  tx_gain_dB = tx_gain_dB         # transmit gain (in dB), beteween -89.75 to 0 dB with a resolution of 0.25 dB
  rx_gain_dB = 40                 # receive gain (in dB), beteween 0 to 74.5 dB (only set if AGC is 'manual')
  rx_agc_mode = 'manual'          # receiver's AGC mode: 'manual', 'slow_attack', or 'fast_attack'
  rx_buffer_size = 200e3          # receiver's buffer size (in samples), length of data returned by sdr.rx()
  tx_cyclic_buffer = True         # cyclic nature of transmitter's buffer (True -> continuously repeat transmission)

  # ---------------------------------------------------------------
  # Initialize Pluto object using issued token.
  # ---------------------------------------------------------------
  sdr = adi.Pluto(token='yhUpi2hcH0k') # create Pluto object
  sdr.sample_rate = int(sample_rate)   # set baseband sampling rate of Pluto

  # ---------------------------------------------------------------
  # Setup Pluto's transmitter.
  # ---------------------------------------------------------------
  sdr.tx_destroy_buffer()                   # reset transmit data buffer to be safe
  sdr.tx_rf_bandwidth = int(tx_rf_bw_Hz)    # set transmitter RF bandwidth
  sdr.tx_lo = int(tx_carrier_freq_Hz)       # set carrier frequency for transmission
  sdr.tx_hardwaregain_chan0 = tx_gain_dB    # set the transmit gain
  sdr.tx_cyclic_buffer = tx_cyclic_buffer   # set the cyclic nature of the transmit buffer

  # ---------------------------------------------------------------
  # Setup Pluto's receiver.
  # ---------------------------------------------------------------
  sdr.rx_destroy_buffer()                   # reset receive data buffer to be safe
  sdr.rx_lo = int(rx_carrier_freq_Hz)       # set carrier frequency for reception
  sdr.rx_rf_bandwidth = int(sample_rate)    # set receiver RF bandwidth
  sdr.rx_buffer_size = int(rx_buffer_size)  # set buffer size of receiver
  sdr.gain_control_mode_chan0 = rx_agc_mode # set gain control mode
  sdr.rx_hardwaregain_chan0 = rx_gain_dB    # set gain of receiver

  # ---------------------------------------------------------------
  # Create transmit signal.
  # ---------------------------------------------------------------
  N_syms = 500     # generate N random M-QAM symbols
  N_zc = 389
  N_pilots = 40     
  M = M             # modulation order
  beta = 0.5        # roll-off factor
  q = 2             # ZC sequence root value
  span = 8
  pulse_shape = mapping.get_rrc_pulse(beta=beta, span=span, sps=sps)

  # generate N M-QAM symbols
  symbols, constellation = mapping.gen_rand_qam_symbols(N_syms, M)
  
  # append N/10 zeros at the end of the sequence of symbols
  symbols_terminated = np.append(symbols, np.zeros(shape=(1, int(N_syms/10))))

  # prepend pilot sequence
  pilot_sequence = np.ones(N_pilots, dtype=np.complex64)
  pilot_sequence = pilot_sequence * (1 + 1j)
  symbols_terminated = np.concatenate((pilot_sequence, symbols_terminated), axis=0)

  # prepend ZC sequence
  zc_sequence = sync.gen_zadoff_chu_sequence(q, N_zc)
  moose_zc_sequence = np.concatenate((zc_sequence, zc_sequence), axis=0)
  # np.save('moose_zc_sequence.npy', moose_zc_sequence)
  zc_symbols_terminated = np.concatenate((moose_zc_sequence, symbols_terminated), axis=0)

  # prepare the zc pulse shaped
  # zc_pulse_train = mapping.create_pulse_train(zc_sequence, sps)
  # zc_pulse_tx_shaped = np.convolve(zc_pulse_train, pulse_shape, mode='full')
  # # zc_pulse_tx_rx_shaped = np.convolve(zc_pulse_tx_shaped, pulse_shape, mode='same')
  
  # create pulse train
  pulse_train = mapping.create_pulse_train(zc_symbols_terminated, sps)
  tx_signal = np.convolve(pulse_train, pulse_shape, mode='full')
  # np.save("tx_signal.npy", tx_signal)

  t = np.arange(0, len(tx_signal))
  fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))
  fig.suptitle("rrc pulse shaped transmitted signal".format(beta))
  axs[0].plot(t, tx_signal.real)
  markerline, stemlines, baseline = axs[0].stem(np.arange(start=0, step=sps, stop=(int(span/2)+N_zc)*sps) ,tx_signal.real[::sps][:(int(span/2)+N_zc)], markerfmt='ro', linefmt='r', label='First ZC')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)
  markerline, stemlines, baseline = axs[0].stem(np.arange(start=(int(span/2)+N_zc)*sps, step=sps, stop=(int(span/2)+2*N_zc)*sps) ,tx_signal.real[::sps][(int(span/2)+N_zc):(int(span/2)+2*N_zc)], markerfmt='bo', linefmt='b', label='Moose Method ZC')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)
  axs[0].axhline(0, color='black', linewidth=0.5)
  axs[0].axvline(0, color='black', linewidth=0.5)
  axs[0].set_title("Pulse train: real")
  axs[0].set_ylabel("Real")
  axs[0].grid(True)
  axs[0].set_xlim([0, 0+1000])
  axs[0].set_ylim([-1.5, 1.5])
  axs[0].legend()

  axs[1].plot(t, tx_signal.imag)
  markerline, stemlines, baseline = axs[1].stem(np.arange(start=0, step=sps, stop=(int(span/2)+N_zc)*sps) ,tx_signal.imag[::sps][:(int(span/2)+N_zc)], markerfmt='ro', linefmt='r', label='First ZC')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)
  markerline, stemlines, baseline = axs[1].stem(np.arange(start=(int(span/2)+N_zc)*sps, step=sps, stop=(int(span/2)+2*N_zc)*sps) ,tx_signal.imag[::sps][(int(span/2)+N_zc):(int(span/2)+2*N_zc)], markerfmt='bo', linefmt='b', label='Moose Method ZC')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)
  axs[1].axhline(0, color='black', linewidth=0.5)
  axs[1].axvline(0, color='black', linewidth=0.5)
  axs[1].set_title("Pulse train: imag")
  axs[1].set_ylabel("Imaginary")
  axs[1].grid(True)
  axs[1].set_xlim([0, 0+1000])
  axs[1].set_ylim([-1.5, 1.5])
  axs[1].legend()
  fig.supxlabel("Symbol Index")
  plt.savefig("figs/rrc_pulse_shaped_transmitted_signal.pdf")
  plt.show()

  # ---------------------------------------------------------------
  # Transmit from Pluto!
  # ---------------------------------------------------------------
  scaling_factor = np.max(np.abs(tx_signal))
  tx_signal_scaled = tx_signal / scaling_factor * 2**14 # Pluto expects TX samples to be between -2^14 and 2^14 
  sdr.tx(tx_signal_scaled) # will continuously transmit when cyclic buffer set to True

  # ---------------------------------------------------------------
  # Receive with Pluto!
  # ---------------------------------------------------------------
  sdr.rx_destroy_buffer() # reset receive data buffer to be safe
  for i in range(1): # clear buffer to be safe
      rx_data_ = sdr.rx() # toss them out
  rx_signal = sdr.rx() # capture raw samples from Pluto

  # rrc pulse shaping at Rx
  rx_signal_filtered = np.convolve(rx_signal, pulse_shape, mode='same') / sps
  # np.save("rx_signal_filterd.npy", rx_signal_filtered)

  # coarse frame synchronization
  correlation = np.correlate(rx_signal_filtered[::sps], moose_zc_sequence)
  begin_index = np.argmax(abs(correlation[:10000])) * sps - int(span/2*sps)
  print("begin index = ", begin_index)

  plt.stem(correlation.real)
  # plt.savefig("correlation_with_ZC_sequence.pdf")
  plt.show()

  ZC_indices = np.arange(start=begin_index, stop=begin_index+2*N_zc*sps, step=sps) 
  pilot_indices = np.arange(start=begin_index+2*N_zc*sps, stop=begin_index+(2*N_zc+N_pilots)*sps, step=sps) 

  # Extract 1 chunk of ZC + Pilots + data
  chunk_indices = np.arange(start=begin_index, stop=begin_index+(2*N_zc+N_pilots+N_syms)*sps, step=sps)
  upsampled_chunk = rx_signal_filtered[np.arange(start=begin_index, stop=begin_index+(int(span/2)+2*N_zc+N_pilots+N_syms)*sps, step=1)]
  print("upsampled_chunk shape: ", upsampled_chunk.shape)

  t = np.arange(0, len(rx_signal_filtered))
  fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))
  fig.suptitle("Rx Matched Filter Output".format(beta))
  axs[0].plot(t, rx_signal_filtered.real)
  markerline, stemlines, baseline = axs[0].stem(chunk_indices ,rx_signal_filtered.real[chunk_indices], markerfmt='ro', linefmt='r')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)

  markerline, stemlines, baseline = axs[0].stem(ZC_indices ,rx_signal_filtered.real[ZC_indices], markerfmt='bo', linefmt='b', label='ZC sequence')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)

  markerline, stemlines, baseline = axs[0].stem(pilot_indices ,rx_signal_filtered.real[pilot_indices], markerfmt='go', linefmt='g', label='Pilot Symbols')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)
  axs[0].axhline(0, color='black', linewidth=0.5)
  axs[0].axvline(0, color='black', linewidth=0.5)
  axs[0].set_title("Pulse train: real")
  axs[0].set_ylabel("Real")
  axs[0].set_xlim([begin_index, begin_index+1000])
  axs[0].set_xticks(np.arange(0, 1000, 200))
  axs[0].grid(True)


  axs[1].plot(t, rx_signal_filtered.imag)
  markerline, stemlines, baseline = axs[1].stem(chunk_indices ,rx_signal_filtered.imag[chunk_indices], markerfmt='ro', linefmt='r')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)

  markerline, stemlines, baseline = axs[1].stem(ZC_indices ,rx_signal_filtered.imag[ZC_indices], markerfmt='bo', linefmt='b', label='ZC sequence')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)

  markerline, stemlines, baseline = axs[1].stem(pilot_indices ,rx_signal_filtered.imag[pilot_indices], markerfmt='go', linefmt='g', label='Pilot Symbols')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)
  axs[1].axhline(0, color='black', linewidth=0.5)
  axs[1].axvline(0, color='black', linewidth=0.5)
  axs[1].set_title("Pulse train: imag")
  axs[1].set_ylabel("Imaginary")
  axs[1].grid(True)
  axs[1].set_xlim([begin_index, begin_index+1000])
  axs[1].set_xticks(ticks=np.arange(start=begin_index, stop=begin_index+1000, step=200), labels=np.arange(start=begin_index, stop=begin_index+1000, step=200)-begin_index)
  plt.savefig("Rx Matched Filter Output.pdf")
  plt.show()


  # symbol synchronization
  tau_d = sync.coarse_symbol_sync(sps, upsampled_chunk)
  print("tau_d = ", tau_d)
  chunk_indices_symbol_synchronized = np.arange(start=0, stop=(2*N_zc+N_pilots+N_syms)*sps, step=sps) + tau_d
  symbol_synched_chunk = upsampled_chunk[tau_d:tau_d+(int(span/2)+2*N_zc+N_pilots+N_syms)*sps:1]
  print("symbol_synched_chunk shape: ", symbol_synched_chunk.shape)

  t = np.arange(0, len(upsampled_chunk))
  fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))
  fig.suptitle("Coarse Symbol Synchronization Output".format(beta))
  axs[0].plot(t, upsampled_chunk.real)
  markerline, stemlines, baseline = axs[0].stem(chunk_indices_symbol_synchronized, upsampled_chunk.real[chunk_indices_symbol_synchronized], markerfmt='ro', linefmt='r')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)

  # markerline, stemlines, baseline = axs[0].stem(ZC_indices ,upsampled_chunk.real[ZC_indices], markerfmt='bo', linefmt='b', label='ZC sequence')
  # markerline.set_markerfacecolor('none')
  # baseline.set_visible(False)

  # markerline, stemlines, baseline = axs[0].stem(pilot_indices ,upsampled_chunk.real[pilot_indices], markerfmt='go', linefmt='g', label='Pilot Symbols')
  # markerline.set_markerfacecolor('none')
  # baseline.set_visible(False)
  axs[0].axhline(0, color='black', linewidth=0.5)
  axs[0].axvline(0, color='black', linewidth=0.5)
  axs[0].set_title("Pulse train: real")
  axs[0].set_ylabel("Real")
  axs[0].set_xlim([0, 0+1000])
  axs[0].grid(True)


  axs[1].plot(t, upsampled_chunk.imag)
  markerline, stemlines, baseline = axs[1].stem(chunk_indices_symbol_synchronized ,upsampled_chunk.imag[chunk_indices_symbol_synchronized], markerfmt='go', linefmt='g')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)

  # markerline, stemlines, baseline = axs[1].stem(ZC_indices ,rx_signal_filtered.imag[ZC_indices], markerfmt='bo', linefmt='b', label='ZC sequence')
  # markerline.set_markerfacecolor('none')
  # baseline.set_visible(False)

  # markerline, stemlines, baseline = axs[1].stem(pilot_indices ,rx_signal_filtered.imag[pilot_indices], markerfmt='go', linefmt='g', label='Pilot Symbols')
  # markerline.set_markerfacecolor('none')
  # baseline.set_visible(False)
  axs[1].axhline(0, color='black', linewidth=0.5)
  axs[1].axvline(0, color='black', linewidth=0.5)
  axs[1].set_title("Pulse train: imag")
  axs[1].set_ylabel("Imaginary")
  axs[1].grid(True)
  axs[1].set_xlim([0, 0+1000])
  plt.savefig("Symbol Synchronization.pdf")
  plt.show()
  # np.save("symbol_synched_chunk.npy", symbol_synched_chunk)

  # fine frame synchronization
  d = sync.frame_sync(moose_zc_sequence, symbol_synched_chunk[:3*N_zc*sps], sps)
  print("fine frame synchronization, d = ", d)
  fine_frame_synched_chunk = symbol_synched_chunk[d*sps:(d+2*N_zc+N_pilots+N_syms)*sps:1]
  print("fine_frame_synched shape: ", fine_frame_synched_chunk.shape)

  # frequency synchronization
  tr1 = symbol_synched_chunk[d*sps:(d+N_zc)*sps:sps]
  tr2 = symbol_synched_chunk[(d+N_zc)*sps:(d+2*N_zc)*sps:sps]
  Delta, detection_range = sync.frequency_synchronization(tr1, tr2, T)
  print("Delta = ", Delta, "; Delta detection range: ", detection_range)
  t = np.arange(start=0, step=1, stop=len(fine_frame_synched_chunk)) * ts
  frequency_sync_correction_factor = np.exp(1j*2*np.pi*Delta*t)
  frequency_synched_chunk = fine_frame_synched_chunk * frequency_sync_correction_factor

  t = np.arange(0, len(frequency_synched_chunk))
  fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))
  fig.suptitle("Frequency Synchronization Output")
  axs[0].plot(t, frequency_synched_chunk.real)
  markerline, stemlines, baseline = axs[0].stem(t[::sps], frequency_synched_chunk.real[::sps], markerfmt='ro', linefmt='r')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)

  # markerline, stemlines, baseline = axs[0].stem(ZC_indices ,upsampled_chunk.real[ZC_indices], markerfmt='bo', linefmt='b', label='ZC sequence')
  # markerline.set_markerfacecolor('none')
  # baseline.set_visible(False)

  # markerline, stemlines, baseline = axs[0].stem(pilot_indices ,upsampled_chunk.real[pilot_indices], markerfmt='go', linefmt='g', label='Pilot Symbols')
  # markerline.set_markerfacecolor('none')
  # baseline.set_visible(False)
  axs[0].axhline(0, color='black', linewidth=0.5)
  axs[0].axvline(0, color='black', linewidth=0.5)
  axs[0].set_title("Pulse train: real")
  axs[0].set_ylabel("Real")
  axs[0].set_xlim([0, 0+1000])
  axs[0].grid(True)


  axs[1].plot(t, frequency_synched_chunk.imag)
  markerline, stemlines, baseline = axs[1].stem(t[::sps] ,frequency_synched_chunk.imag[::sps], markerfmt='go', linefmt='g')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)

  # markerline, stemlines, baseline = axs[1].stem(ZC_indices ,rx_signal_filtered.imag[ZC_indices], markerfmt='bo', linefmt='b', label='ZC sequence')
  # markerline.set_markerfacecolor('none')
  # baseline.set_visible(False)

  # markerline, stemlines, baseline = axs[1].stem(pilot_indices ,rx_signal_filtered.imag[pilot_indices], markerfmt='go', linefmt='g', label='Pilot Symbols')
  # markerline.set_markerfacecolor('none')
  # baseline.set_visible(False)
  axs[1].axhline(0, color='black', linewidth=0.5)
  axs[1].axvline(0, color='black', linewidth=0.5)
  axs[1].set_title("Pulse train: imag")
  axs[1].set_ylabel("Imaginary")
  axs[1].grid(True)
  axs[1].set_xlim([0, 0+1000])
  plt.savefig("Frequency Scynchronization.pdf")
  plt.show()

  # LS estimation
  y = np.array(frequency_synched_chunk[2*N_zc*sps:(2*N_zc+N_pilots)*sps:sps]).reshape((N_pilots,))
  print("y= ", y)
  # t = pilot_sequence.reshape((N_pilots,))
  # h_est = np.vdot(t, y) / np.vdot(t, t)
  y_avg = np.mean(y)
  h_est = y_avg / (1+1j)
  print("h = ",h_est)

  # equalization
  equalized_chunk = frequency_synched_chunk / h_est

  t = np.arange(0, len(equalized_chunk))
  fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))
  fig.suptitle("Equalized chunk")
  axs[0].plot(t, equalized_chunk.real)
  markerline, stemlines, baseline = axs[0].stem(np.arange(start=0, step=sps, stop=len(t)), equalized_chunk.real[::sps], markerfmt='ro', linefmt='r')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)
  axs[0].axhline(0, color='black', linewidth=0.5)
  axs[0].axvline(0, color='black', linewidth=0.5)
  axs[0].set_title("Pulse train: real")
  axs[0].set_ylabel("Real")
  axs[0].grid(True)
  axs[0].set_xlim([2000, 2000+1000])
  axs[0].set_ylim([-1.5, 1.5])

  axs[1].plot(t, equalized_chunk.imag)
  markerline, stemlines, baseline = axs[1].stem(np.arange(start=0, step=sps, stop=len(t)), equalized_chunk.imag[::sps], markerfmt='bo', linefmt='b')
  markerline.set_markerfacecolor('none')
  baseline.set_visible(False)
  axs[1].axhline(0, color='black', linewidth=0.5)
  axs[1].axvline(0, color='black', linewidth=0.5)
  axs[1].set_title("Pulse train: imag")
  axs[1].set_ylabel("Imaginary")
  axs[1].grid(True)
  axs[1].set_xlim([2000, 2000+1000])
  axs[1].set_ylim([-1.5, 1.5])
  plt.savefig("Equalized_chunk.pdf")
  plt.show()

  # extract data from the chunk
  equalized_data_chunk = equalized_chunk[(2*N_zc+N_pilots)*sps:(2*N_zc+N_pilots+N_syms)*sps:sps]
  data_chunk = zc_symbols_terminated[2*N_zc+N_pilots:2*N_zc+N_pilots+N_syms:1]

  # plotting
  plt.scatter(equalized_data_chunk.real, equalized_data_chunk.imag, marker='o',facecolor='none', edgecolors='red', label='Received Constellation')
  plt.scatter(data_chunk.real, data_chunk.imag, marker='o', c='green', label='64-QAM Constellation')
  plt.axvline(0, color='black', linewidth=0.5)
  plt.axhline(0, color='black', linewidth=0.5)
  plt.xlim([-1.5, 1.5])
  plt.ylim([-1.5, 1.5])
  plt.legend()
  plt.savefig(f"Received_constellation_{CFO}.pdf")
  plt.show()

  # symbol detection
  num_incorrect = 0
  detected_symbol_index = np.zeros(shape=(len(equalized_data_chunk)), dtype=int)
  true_symbol_index = np.zeros(shape=(len(equalized_data_chunk)), dtype=int)
  print("detected symbol size: ", detected_symbol_index.shape)


  # count number of incorrect detection
  for i_raw_symbol in np.arange(0, len(equalized_data_chunk)):
    detected_symbol_index[i_raw_symbol] = int(np.argmin((equalized_data_chunk[i_raw_symbol] - np.array(constellation))**2))
    true_symbol_index[i_raw_symbol] = int(np.argmin((data_chunk[i_raw_symbol] - np.array(constellation))**2))
    num_incorrect += (detected_symbol_index[i_raw_symbol] != true_symbol_index[i_raw_symbol])
  print("num incorrect: ", num_incorrect)
  print("error rate: ", num_incorrect/N_syms)

  # return num_incorrect, N_syms

if __name__ == '__main__':
  main()