import numpy as np

def frequency_synchronization(tr1, tr2, T_s):
  """ performs frequency synchronization using the Moose method.
    Delta, range = frequency_synchronization(N_moose)
    - Input:
      tr1: first group of received downsampled training samples.
      tr2: second group of received downsampled training samples.
      T_s: symbol period in seconds.
    - Output:
      Delta: CFO, f_tx - f_rx in radians.
      range: range of detection granted by N_moose in radians
  """
  N_tr = len(tr1)
  correlation = np.correlate(tr1, tr2)
  phase = np.angle(correlation)
  epsilon = phase / (2*np.pi*N_tr)
  Delta = epsilon / T_s
  range = [-1/(2*N_tr*T_s), 1/(2*N_tr*T_s)]
  return Delta, range

def coarse_symbol_sync(sps, signal):
  """ performs Maximum Output Energy coarse symbol synchronization and finds tau_d in between [-T/2, T/2].
    tau_d = coarse_symbol_sync(signal)
    - Input:
      sps: samples per symbol.
      signal: oversampled signal to find tau_d.
    - Output:
      tau_d: fractional time delay.
  """
  tau_d_candidates = np.arange(start=-sps/2, step=1, stop=sps/2+1)
  output_energy = [np.sum(signal[::i]**2) for i in tau_d_candidates]
  tau_d = tau_d_candidates[np.argmax(output_energy)]
  return tau_d

def gen_zadoff_chu_sequence(q, N_zc):
  """ generates a Zadoff-Chu sequence using the root index q and the length of the sequence N_zc.
    zc_sequence = gen_zadoff_chu_sequence(q, N_zc)
    - Input:
      q: root index ranging from (1, 2, ..., N_zc - 1).
      N_zc: length of the sequence. Must be an odd number, and is often a prime number.
    - Output:
      zc_sequence: the q-th ZC sequence.
  """
  zc_sequence = np.zeros(N_zc, dtype=np.complex64)
  for n in np.arange(stop=N_zc):
    zc_sequence[n] = np.exp(-1j*np.pi*q*n*(n+1)/N_zc)
  return zc_sequence

def frame_sync(zc_sequence, signal):
  """ performs correlation with downsampled ZC sequence and finds d, the integer time offset.
    d = frame_sync(zc_sequence, signal)
    - Input:
      zc_sequence: training Zadoff-Chu sequence.
      signal: a segment of received signal containing only 1 copy of the transmitted signal.
  """
  correlation = np.correlate(signal, zc_sequence)
  d = np.argmax(correlation)
  return d