import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib.pyplot as plt

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

#theta = np.random.normal(scale=0.5,size=11)
#b = theta[0:6];
#a = np.ndarray(6)
#a[0] = 1;
#a[1:7] = theta[6:11]
print(b)
print(a)
b = b + np.random.normal(scale=1e-9, size=len(b));
a = a + np.random.normal(scale=1e-9, size=len(a));
print(b)
print(a)
theta = np.concatenate([b,a[1:7]])
print(theta.shape)
w, h = scipy.signal.freqz(b,a,w);
varP = 10
diffP = np.sum(((20*np.log10(np.abs(h)) - H))**2);
print(diffP)

plt.semilogx(w,20*np.log10(np.abs(h)));
plt.semilogx(w,H);
#plt.pause(1e-3)
plt.show()

numIters = 1000000;
thetaHist = np.zeros([11,numIters])
varHist = np.zeros(numIters)
thetaHist[:,0] = theta;

for i in range(numIters):
	propScale = 1e-10;
	var = varP + np.random.normal(scale=0.1);
	thetaProp = theta + np.concatenate([np.random.normal(scale=propScale,size=6), np.random.normal(scale=propScale,size=5)])
#	while any(np.abs(thetaProp[6:11]) > 15):
#		thetaProp = theta + np.concatenate([np.random.normal(scale=0.0001,size=6), np.random.normal(scale=0.0001,size=5)])
	b = thetaProp[0:6];
	a = np.ndarray(6)
	a[0] = 1;
	a[1:7] = thetaProp[6:11]
	w, h = scipy.signal.freqz(b,a,w);
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

b = [np.mean(thetaHist[0,:]), np.mean(thetaHist[1,:]), np.mean(thetaHist[2,:]), np.mean(thetaHist[3,:]), np.mean(thetaHist[4,:]), np.mean(thetaHist[5,:])];
a = [1, np.mean(thetaHist[6,:]), np.mean(thetaHist[7,:]), np.mean(thetaHist[8,:]), np.mean(thetaHist[9,:]), np.mean(thetaHist[10,:])]

print(b)
print(a)
print(scipy.signal.butter(5, 0.01,btype='highpass'))

w, h = scipy.signal.freqz(b,a,w);
plt.figure(1)
plt.semilogx(w,20*np.log10(abs(h)))
plt.semilogx(w,H)

plt.figure(2);
for i in range(0,5):
	plt.plot(thetaHist[i,:]);
plt.figure(3)
for i in range(6,11):
	plt.plot(thetaHist[i,:])

plt.figure(4)
plt.plot(varHist)
plt.show()
