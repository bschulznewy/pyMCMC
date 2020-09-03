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

numFreqPoints = 100

# Get w in rads so that we don't need to scale later when GPU code is written
w = np.logspace(np.log10(30), np.log10(20000), num=numFreqPoints)/24000*np.pi;

w, h = scipy.signal.freqz(IR,1,w)

H = 20*np.log10(np.abs(h));
H = H - np.max(np.abs(H))

b,a = scipy.signal.butter(5, 0.01, btype='highpass');
print(b)
print(a)
w, h = scipy.signal.freqz(b,a,w)

H = 20*np.log10(np.abs(h)) + np.random.normal(scale=5, size=len(h));

freq = 0.01;
gain = -5;
Q = 1;
b, a = biquad.peaking(freq, gain, Q=Q);
theta = np.zeros(6)
theta[0] = freq;
theta[1] = gain;
theta[2] = Q;
theta[3] = freq;
theta[4] = gain;
theta[5] = Q;
w, h = scipy.signal.freqz(b,a,w);
h = h*h; # just init with 2 systems on top of each other.
varP = 10
diffP = np.sum(((20*np.log10(np.abs(h)) - H))**2);
print(diffP)

plt.semilogx(w,20*np.log10(np.abs(h)));
plt.semilogx(w,H);
#plt.pause(1e-3)
plt.show()

numIters = 1000000;
thetaHist = np.zeros([6,numIters])
varHist = np.zeros(numIters)
thetaHist[:,0] = theta;

sigma = np.zeros(6);
sigma[0] = 0.001;
sigma[1] = 0.1;
sigma[2] = 0.01;
sigma[3] = 0.001;
sigma[4] = 0.1;
sigma[5] = 0.01;

for i in range(numIters):
	propScale = 5e-4;
	var = varP + np.random.normal(scale=0.1);
	tmp = np.random.normal(size=6);
	thetaProp = theta + tmp*sigma;
	# Force positive frequencies
	while thetaProp[0] < 0 or thetaProp[3] < 0:
		tmp = np.random.normal(size=6);
		thetaProp = theta + tmp*sigma;
	b,a = biquad.peaking(thetaProp[0], thetaProp[1], thetaProp[2]);
	w, h1 = scipy.signal.freqz(b,a,w);
	b,a = biquad.peaking(thetaProp[3], thetaProp[4], thetaProp[5]);
	w, h2 = scipy.signal.freqz(b,a,w);
	h = h1*h2;
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

thetaMean = np.mean(thetaHist,axis=1)

print(thetaMean);

b,a = biquad.peaking(thetaMean[0], thetaMean[1], thetaMean[2]);
w, h1 = scipy.signal.freqz(b,a,w);
b,a = biquad.peaking(thetaMean[3], thetaMean[4], thetaMean[5]);
w, h2 = scipy.signal.freqz(b,a,w);
h = h1*h2;
plt.figure(1)
plt.semilogx(w,20*np.log10(abs(h)))
plt.semilogx(w,H)

for i in range(0,2):
	plt.figure(i+2);
	plt.plot(thetaHist[i,:]);

plt.figure(5)
plt.plot(varHist)
plt.show()
