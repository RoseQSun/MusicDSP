import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import convolve
import time


# Question 1
def myTimeConv(x,h):
    leny = len(x)+ len(h)- 1
    y = np.zeros(leny)
    m = len(x)- 1
    n = len(h)- 1
    for i in range(leny):
        low = max(0, i- n)
        high = min(m, i)
        for j in np.arange(low,high+1,1):
            y[i]= x[j]* h[i-j]+ y[i]
    return(y)

# 200 + 100 - 1 = 299

x = np.ones(200)
h = np.concatenate((np.linspace(0, 1, 26), np.linspace(1, 0, 26)[1:]), axis=None)
y_time = myTimeConv(x, h)
# print(y_time)
plt.plot(y_time)
plt.xlabel('Sample')
plt.ylabel('Convolution')
plt.title('Discrete Time Domain Convolution')
plt.savefig('011-convolution.png')


# Question 2
def loadSoundFile(filename):
    fs, x = read(filename)
    # if x.shape[1] > 1:
    #     x = x[:,0]
    x = np.array(x,dtype = float)
    return (x)

def CompareConv(x,h):
    times = np.zeros(2)
    start1 = time.time()
    spconv = convolve(x,h)
    stop1 = time.time()
    times[0] = stop1-start1
    start2 = time.time()
    myconv = myTimeConv(x,h)
    stop2 = time.time()
    times[1] = stop2-start2

    diff = spconv- myconv
    m = np.mean(diff)
    mabs = np.mean(np.abs(diff))
    stdev = np.std(diff)
    return(m,mabs,stdev,times)



x = loadSoundFile('piano.wav')
h = loadSoundFile('impulse-response.wav')
compare = CompareConv(x,h)
print(compare)