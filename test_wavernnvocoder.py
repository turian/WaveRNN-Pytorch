import sys
sys.path.insert(0,'lib/build-src-RelDebInfo')
import WaveRNNVocoder
import numpy as np

a=WaveRNNVocoder.Vocoder()

a.loadWeights('model_outputs/model.bin')

mel_file='../TrainingData/LJSpeech-1.0.wavernn/mel/00001.npy'
mel = np.load(mel_file)
mel = mel.astype('float32')
wav=a.melToWav(mel)
print()