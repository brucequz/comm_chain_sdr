import numpy as np



def gen_rand_qam_symbols(N, M=4):
  """ Generates N random symbols from a square M-QAM constellation normalized to unit average symbol energy.
    symbols, constellation = gen_rand_qam_symbols(N, M=4)
    - Input:
        N: Number of random symbols to generate.
        M: Order of the QAM constellation (e.g., 4 for QPSK, 16 for 16-QAM).
    - Output:
        symbols: N randomly selected M-QAM symbols.
        constellation: The full M-QAM constellation.
  """
  
  A = np.sqrt(3 / (2 * (M - 1)))
  amplitudes = [A*(2*m+1-np.sqrt(M)) for m in np.arange(0, np.sqrt(M))]
  constellation = [i+1j*j for i in amplitudes for j in amplitudes]

  # generate random number as indices to pick constellation point
  symbol_indices = np.random.randint(low=0, high=len(constellation), size=N)
  symbols = np.array(constellation)[symbol_indices]
  return symbols, constellation

  
def create_pulse_train(symbols, sps):
  """ Forms a pulse train from a sequence of symbols by inserting zeros between the symbols.
    pulse_train = create_pulse_train(symbols, sps)
    - Input: 
      symbols: The QAM symbols from above.
      sps: Samples per symbol, i.e., the number of discrete-time samples from one symbol to the next.
    - Output:
      pulse_train: A discrete-time signal where each symbol is separated by (sps-1) zeros.
    - Note: The zeroth element of the output pulse train should be the zeroth symbol (not a zero).
  """

  num_symbols = len(symbols)
  pulse_train = np.zeros(num_symbols + (num_symbols) * (sps-1), dtype=complex)
  pulse_train[::sps] = symbols
  return pulse_train
    

def get_rc_pulse(beta, span, sps):
  """ Generates a raised cosine pulse shape.
    pulse = get_rc_pulse(beta, span, sps)
    - Input:
      beta: Rolloff factor 𝛽 (between 0 and 1, inclusive).
      span: The integer number of symbol durations spanned by the pulse, not including the symbol at 𝑡=0
      sps: Samples per symbol.
    - Output:
      pulse: A raised cosine pulse (normalized such that its peak value is unity), symmetric and centered at 𝑡=0
      . The number of zero crossings should be equal to span.
  """
  T = 1
  t = np.linspace(-span/2, span/2, span*sps+1)
  pulse = np.zeros_like(t)

  # beta = 0
  if beta == 0.0:
    pulse = np.sinc(t/T) / T
    return pulse
  
  # beta none 0
  for i_t in range(len(t)):
    if t[i_t] == 1/(2*beta) or t[i_t] == -1/(2*beta):
      pulse[i_t] = np.pi * np.sinc(1/(2*beta)) / (4 * T)
    else:
      pulse[i_t] = np.sinc(t[i_t] / T) * np.cos(np.pi * beta * t[i_t] / T) / (T * (1 - (2*beta*t[i_t]/T)**2))

  return pulse

def get_rrc_pulse(beta, span, sps):
  """ Generates a root raised cosine pulse shape.
    pulse = get_rrc_pulse(beta, span, sps)
    - Input:
      beta: Rolloff factor 𝛽 (between 0 and 1, inclusive).
      span: The integer number of symbol durations spanned by the pulse, not including the symbol at 𝑡=0
      sps: Samples per symbol.
    - Output:
      pulse: A root raised cosine pulse (normalized such that its peak value is unity), symmetric and centered at 𝑡=0
      . The number of zero crossings should be equal to span.
  """
  T = 1
  t = np.linspace(-span/2, span/2, span*sps+1)
  pulse = np.zeros_like(t)


  for i_t in range(len(t)):
    if t[i_t] == 0:
      pulse[i_t] = (1+beta*(4/np.pi-1)) / T
    elif beta != 0 and (t[i_t] == T/(4*beta) or t[i_t] == -T/(4*beta)):
      pulse[i_t] = beta*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta))) / (np.sqrt(2)*T)
    else:
      pulse[i_t] = (1/T)*(np.sin(np.pi*t[i_t]*(1-beta)/T)+4*beta*t[i_t]*np.cos(np.pi*t[i_t]*(1+beta)/T)/T) / (np.pi*t[i_t]*(1-(4*beta*t[i_t]/T)**2)/T)

  return pulse