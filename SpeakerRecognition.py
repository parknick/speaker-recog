# %% General Imports

import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as speech

from sklearn.mixture import GaussianMixture

# %% Read in training and testing data and create two data sets

NUM_TRAINING_SAMPLES = 3
NUM_TESTING_SAMPLES = 3

# Each entry in the training and testing sets will containg a tuple consisting of the
# sample rate of the wav file and a numpy array containing the data read from the wav file
training_set = []
testing_set = []
for i in range(1, NUM_TRAINING_SAMPLES + 1):
    training_set.append(wav.read('./wav-files/0%d_train.wav' % i))
    testing_set.append(wav.read('./wav-files/0%d_test.wav' % i))

#-------------------------------------- END SECTION --------------------------------------#

# %% Perform feature extraction on the data

training_features = []
testing_features = []
for i in range(NUM_TRAINING_SAMPLES):
    training_features.append(speech.mfcc(training_set[i][1], training_set[i][0]))
    testing_features.append(speech.mfcc(testing_set[i][1], testing_set[i][0]))

#-------------------------------------- END SECTION --------------------------------------#

# %% Train the GMM using the training samples

#-------------------------------------- END SECTION --------------------------------------#

# %% Perform classification using the testing data

# TODO Record all the classification scores and generate a confusion matrix

#-------------------------------------- END SECTION --------------------------------------#

# %% Perform validation on a probe sample (not in the database) and match it against the trained model

# TODO Record the classification score

#-------------------------------------- END SECTION --------------------------------------#

# %% Evaluation

# TODO Evaluate the error rates using the confusion matrix

#-------------------------------------- END SECTION --------------------------------------#
