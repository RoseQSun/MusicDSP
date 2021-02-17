import numpy as np
import matplotlib.pyplot as plt


# Question 1 
def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    amplitude = float(amplitude)
    float(phase_radians)
    sampling_rate_Hz = float(sampling_rate_Hz) if sampling_rate_Hz >= 2*frequency_Hz else False
    length_secs = float(length_secs) if length_secs > 0 else False
    t = np.arange(0,length_secs+1/sampling_rate_Hz,1/sampling_rate_Hz)
    x = amplitude * np.sin(2*np.pi*frequency_Hz*t + phase_radians)
    return(t,x)

(t1,x1) = generateSinusoidal(amplitude = 1.0, sampling_rate_Hz = 44100, frequency_Hz = 400, length_secs = 0.5, phase_radians = np.pi/2)
n = round(44100 * 0.005)
plt.plot(t1[:n],x1[:n])
plt.title('Generated Sine Wave')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.savefig('01-sinewave.png')


# Question 2
def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    frequencies = [(frequency_Hz * i) for i in range(1,20,2)]
    amplitudes = [(amplitude / j) for j in range(1,20,2)]
    xsine = np.vstack([generateSinusoidal(amplitude = j, sampling_rate_Hz = sampling_rate_Hz, frequency_Hz = i, length_secs = length_secs, phase_radians = phase_radians)[1] for (i,j) in zip(frequencies,amplitudes)])
    x = 4/np.pi* np.sum(xsine, axis=0)
    t = np.arange(0,length_secs,1/sampling_rate_Hz)
    return(t,x)

(t2,x2) = generateSquare(amplitude = 1.0, sampling_rate_Hz = 44100, frequency_Hz = 400, length_secs = 0.5, phase_radians = 0)
plt.figure()
plt.plot(t2[:n],x2[:n])
plt.title('Generated Square Wave')
plt.xlabel('Time(sec)')
plt.ylabel('Amplitude')
plt.savefig('02-squarewave.png')


# Question 3
def computeSpectrum(x, sample_rate_Hz):
    fft= np.fft.fft(x)
    X = fft[:round(len(x)/2)]
    f = np.linspace(0,sample_rate_Hz*0.5,len(X))
    XAbs = np.abs(X)
    XPhase = np.angle(X)
    XRe = np.real(X)
    XIm = np.imag(X)
    return (f, XAbs, XPhase, XRe, XIm)

sine_f, sine_XAbs, sine_XPhase, sine_XRe, sine_XIm = computeSpectrum(x1, 44100)
square_f, square_XAbs, square_XPhase, square_XRe, square_XIm = computeSpectrum(x2, 44100)

plt.figure()
plt.subplot(2,1,1)
plt.title('Sine Wave FFT')
plt.plot(sine_f, sine_XAbs)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.subplot(2,1,2)
plt.plot(sine_f, sine_XPhase)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (rad)')
plt.savefig('03-SineFFT.png')

plt.figure()
plt.subplot(2,1,1)
plt.title('Square Wave FFT')
plt.plot(square_f, square_XAbs)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.subplot(2,1,2)
plt.plot(square_f, square_XPhase)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (rad)')
plt.savefig('03-SquareFFT.png')


# Question 4
def computeNewSpectrum(x, sample_rate_Hz, window_type):
    fft= np.fft.fft(x)
    X_unwin = fft[:round(len(x)/2)]
    if window_type == 'rect':
        rect = np.ones(round(len(x)/2))
        X = rect * X_unwin
    elif window_type == 'hann':
        hann = np.hanning(round(len(x)/2))
        # plt.plot(hann)
        # plt.savefig('hann.png')
        X = hann * X_unwin
    f = np.linspace(0,sample_rate_Hz*0.5,len(X))
    XAbs = np.abs(X)
    XPhase = np.angle(X)
    XRe = np.real(X)
    XIm = np.imag(X)
    return (f, XAbs, XPhase, XRe, XIm)


rect_f, rect_XAbs, rect_XPhase, rect_XRe, rect_XIm = computeNewSpectrum(x2, 44100, 'rect')
hann_f, hann_XAbs, hann_XPhase, hann_XRe, hann_XIm = computeNewSpectrum(x2, 44100, 'hann')

plt.figure()
plt.title('Square Wave Rectangular Window')
plt.plot(rect_f, rect_XAbs)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.savefig('04-SquareRect.png')

plt.figure()
plt.title('Square Wave Hanning Window')
plt.plot(hann_f, hann_XAbs)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.savefig('04-SquareHann.png')


# Question 5
def linearchirp(amplitude, fs, length, starting_f, stopping_f, phase):
    t = np.arange(0, length, 1/fs)
    ft = starting_f + (stopping_f - starting_f) * t /length
    chirp = amplitude * np.cos(2*np.pi*ft*t+phase)
    return(chirp)

plt.figure()
chirps = linearchirp(1, 8000, 1.5, 100, 1000, 0)
plt.subplot(2,1,1)
plt.plot(chirps[:2000])
plt.subplot(2,1,2)
plt.plot(abs(np.fft.fft(chirps)))
plt.savefig('05-LinearChirp')