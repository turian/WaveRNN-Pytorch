import sys
import glob

from scipy.io.wavfile import write


sys.path.insert(0,'lib/build-src-RelDebInfo')
sys.path.insert(0,'library/build-src-Desktop-RelWithDebInfo')
import WaveRNNVocoder
import numpy as np

vocoder=WaveRNNVocoder.Vocoder()

vocoder.loadWeights('model_outputs/model.bin')

# mel_file='../TrainingData/LJSpeech-1.0.wavernn/mel/00001.npy'
# mel1 = np.load(mel_file)
# mel1 = mel1.astype('float32')
# wav=vocoder.melToWav(mel)
# print()

filelist = glob.glob('eval/mel*.npy')

for fname in filelist:
    mel = np.load(fname).T
    wav = vocoder.melToWav(mel)
    break

#scaled = np.int16(wav/np.max(np.abs(wav)) * 32767)
write('test.wav',16000, wav)

print()

fnames=['inputs/00000.npy','inputs/mel-northandsouth_01_f000001.npy']
mel0=np.load(fnames[0])
mel1=np.load(fnames[1]).T
mel2=np.load(filelist[0]).T

