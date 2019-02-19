import sys
import glob

sys.path.insert(0,'lib/build-src-RelDebInfo')
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

print()