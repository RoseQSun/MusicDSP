import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import correlate, find_peaks

# Question 1

def crossCorr(x,y):
    z = correlate(x,y)
    return (z)

def loadSoundFile(filename):
    fs, x = read(filename)
    if x.shape[1] > 1:
        x = x[:,0]
    x = np.array(x,dtype = float)
    return (x)

def main(snareFilename, drumloopFilename):
    x = loadSoundFile(snareFilename)
    y = loadSoundFile(drumloopFilename)
    z = crossCorr(x,y)
    plt.plot(z)
    plt.title('Snare and Drum_loop Correlation')
    plt.xlabel('Sample')
    plt.ylabel('Correlation')
    plt.savefig('01-correlation.png')
    
main('snare.wav','drum_loop.wav')
    
# Question 2

def findSnarePosition(snareFilename, drumloopFilename):
    x = loadSoundFile(snareFilename)
    y = loadSoundFile(drumloopFilename)
    z = crossCorr(x,y)
    pos, _ = find_peaks(z,height = 1.75e11)
    return(pos)
    
print(findSnarePosition('snare.wav','drum_loop.wav'))