import sys
import glob

from scipy.io.wavfile import write
from scipy import signal

sys.path.insert(0,'lib/build-src-RelDebInfo')
sys.path.insert(0,'library/build-src-Desktop-RelWithDebInfo')
import WaveRNNVocoder
import numpy as np
import time
import pylab as plt
from hparams import hparams as hp

vocoder=WaveRNNVocoder.Vocoder()

vocoder.loadWeights('model_outputs/model.bin')

# mel_file='../TrainingData/LJSpeech-1.0.wavernn/mel/00001.npy'
# mel1 = np.load(mel_file)
# mel1 = mel1.astype('float32')
# wav=vocoder.melToWav(mel)
# print()

filelist = glob.glob('eval/mel*.npy')

k=0.97
for fname in filelist:

    mel = np.load(fname).T
    #vocoder.setDebugLevel(0)
    t_1 = time.time()
    wav = vocoder.melToWav(mel)
    print(" >  {} N mels {} Run-time: {}".format(fname, mel.shape[1], time.time() - t_1))
    # vocoder.setDebugLevel(1)
    # wav1 = vocoder.melToWav(mel)
    # vocoder.setDebugLevel(2)
    # wav2 = vocoder.melToWav(mel)
    #wav1 = signal.lfilter([1], [1, -k], wav)
    write(fname+'.wav', 16000, wav)
    break

#scaled = np.int16(wav/np.max(np.abs(wav)) * 32767)
write('test.wav',16000, wav)

print()

fnames=['inputs/00000.npy','inputs/mel-northandsouth_01_f000001.npy']
mel0=np.load(fnames[0])
# mel1=np.load(fnames[1]).T
# mel2=np.load(filelist[0]).T



from scipy import signal
firwin = signal.firwin(hp.n_fft, [75, 7600], pass_zero=False, fs=hp.sample_rate)



