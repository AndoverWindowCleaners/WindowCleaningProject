import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,4*np.pi,200)
y = np.abs(np.cos(x))**2
plt.plot(x,y)

coeffs = np.fft.rfft(y,200)
amps = 2*np.abs(coeffs)/200

freqs = np.arange(len(amps))
plt.plot(freqs, amps)
plt.show()