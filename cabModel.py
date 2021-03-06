import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt
import biquad

fs, inWav = scipy.io.wavfile.read('input.wav');
fs, outWav = scipy.io.wavfile.read('output.wav');

inWav = inWav[0:48000].astype('float32');
outWav = outWav[0:48000].astype('float32');

IR = np.convolve(np.flip(inWav), outWav, mode='same');
IR = IR[25330:40000];

numFreqPoints = 1000

# Get w in rads so that we don't need to scale later when GPU code is written
w = np.logspace(np.log10(30), np.log10(20000), num=numFreqPoints)/24000*np.pi;

w, h = scipy.signal.freqz(IR,1,w)

H = 20*np.log10(np.abs(h));
H = H - np.max(np.abs(H)) + 10

# b,a = scipy.signal.butter(5, 0.01, btype='highpass');
# w, h = scipy.signal.freqz(b,a,w)

# H = 20*np.log10(np.abs(h)) + np.random.normal(scale=5, size=len(h));

freq = 1000/24000;
gain = 0;
Q = 1;
theta = np.zeros(10) # w and Q for high/low pass plus 2x biquads

# Highpass
theta[0] = 100/24000; # scaled [0:1]
theta[1] = 1; # Q
# Lowpass
theta[2] = 5000/24000; # 5kHz cutoff
theta[3] = 1; # Q
# Peaking 1
theta[4] = freq;
theta[5] = gain;
theta[6] = Q;
# Peaking 2
theta[7] = freq;
theta[8] = gain;
theta[9] = Q;

b, a = biquad.highpass(theta[0], theta[1]);
w, h1 = scipy.signal.freqz(b,a,w);
b, a = biquad.lowpass(theta[2], theta[3]);
w, h2 = scipy.signal.freqz(b,a,w);
b, a = biquad.peaking(theta[4], theta[5], theta[6]);
w, h3 = scipy.signal.freqz(b,a,w);
b, a = biquad.peaking(theta[7], theta[8], theta[9]);
w, h4 = scipy.signal.freqz(b,a,w);

h = h1*h2*h3*h4; # just init with 2 systems on top of each other.
varP = 10
diffP = np.sum(((20*np.log10(np.abs(h)) - H))**2);
print(diffP)

plt.semilogx(w,20*np.log10(np.abs(h)));
plt.semilogx(w,H);
#plt.pause(1e-3)
plt.show()

numIters = 1000000;
thetaHist = np.zeros([10,numIters])
varHist = np.zeros(numIters)
thetaHist[:,0] = theta;

sigma = np.zeros(10);
sigma[0] = 0.001;
sigma[1] = 0.1;
sigma[2] = 0.1;
sigma[3] = 0.1;
sigma[4] = 0.001;
sigma[5] = 0.1;
sigma[6] = 0.01;
sigma[7] = 0.001;
sigma[8] = 0.1;
sigma[9] = 0.01;

sigma = sigma*0.2;

for i in range(numIters):
	propScale = 5e-4;
	var = varP + np.random.normal(scale=1)
	tmp = np.random.normal(size=10);
	thetaProp = theta + tmp*sigma;
	# Force positive frequencies
	while thetaProp[0] < 0 or thetaProp[2] < 0 or thetaProp[4] < 0 or thetaProp[7] < 0:
		tmp = np.random.normal(size=10);
		thetaProp = theta + tmp*sigma;
	
	b, a = biquad.highpass(thetaProp[0], thetaProp[1]);
	w, h1 = scipy.signal.freqz(b,a,w);
	b, a = biquad.lowpass(thetaProp[2], thetaProp[3]);
	w, h2 = scipy.signal.freqz(b,a,w);
	b, a = biquad.peaking(thetaProp[4], thetaProp[5], thetaProp[6]);
	w, h3 = scipy.signal.freqz(b,a,w);
	b, a = biquad.peaking(thetaProp[7], thetaProp[8], thetaProp[9]);
	w, h4 = scipy.signal.freqz(b,a,w);

	h = h1*h2*h3*h4; 

	diff = np.sum(((20*np.log10(np.abs(h)) - H))**2);
	P = len(h)*np.log(varP/var) - 1/(2*var)*diff + 1/(2*varP)*diffP
	print(P, diff, diffP)
	if P > np.log(np.random.randn()):
		theta = thetaProp
		diffP = diff
		varP = var

	print(i/numIters)
	thetaHist[:,i] = theta
	varHist[i] = varP

thetaProp = np.mean(thetaHist,axis=1)

print(thetaProp);
b, a = biquad.highpass(thetaProp[0], thetaProp[1]);
w, h1 = scipy.signal.freqz(b,a,w);
b, a = biquad.lowpass(thetaProp[2], thetaProp[3]);
w, h2 = scipy.signal.freqz(b,a,w);
b, a = biquad.peaking(thetaProp[4], thetaProp[5], thetaProp[6]);
w, h3 = scipy.signal.freqz(b,a,w);
b, a = biquad.peaking(thetaProp[7], thetaProp[8], thetaProp[9]);
w, h4 = scipy.signal.freqz(b,a,w);

h = h1*h2*h3*h4; 

plt.figure(1)
plt.semilogx(w,20*np.log10(abs(h)))
plt.semilogx(w,H)

for i in range(0,2):
	plt.figure(i+2);
	plt.plot(thetaHist[i,:]);

plt.figure(5)
plt.plot(varHist)
plt.show()
