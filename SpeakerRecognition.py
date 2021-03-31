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

for i in range(training_set[0][1].size):
    print(training_set[0][1][i])
#-------------------------------------- END SECTION --------------------------------------#

# %% Perform feature extraction on the data

training_features = []
testing_features = []
for i in range(NUM_TRAINING_SAMPLES):
    training_features.append(speech.mfcc(training_set[i][1], training_set[i][0]))
    testing_features.append(speech.mfcc(testing_set[i][1], testing_set[i][0]))

#-------------------------------------- END SECTION --------------------------------------#

# %% Train the GMM models using the training samples
NUM_G_COMPONENTS = 10

gmm = []
for i in range(NUM_TRAINING_SAMPLES):
    gmm.append(GaussianMixture(n_components=NUM_G_COMPONENTS).fit(training_features[i]))

#-------------------------------------- END SECTION --------------------------------------#

# %% Perform classification using the testing data
c_mat = np.zeros((3,3))

test1_scores = []
for i in range(len(gmm)):
    test1_scores.append(gmm[i].score(testing_features[0]))

test2_scores = []
for i in range(len(gmm)):
    test2_scores.append(gmm[i].score(testing_features[1]))

test3_scores = []
for i in range(len(gmm)):
    test3_scores.append(gmm[i].score(testing_features[2]))

c_mat[0][0] = np.mean(test1_scores[0])
c_mat[1][0] = np.mean(test2_scores[0])
c_mat[2][0] = np.mean(test3_scores[0])

c_mat[0][1] = np.mean(test1_scores[1])
c_mat[1][1] = np.mean(test2_scores[1])
c_mat[2][1] = np.mean(test3_scores[1])

c_mat[0][2] = np.mean(test1_scores[2])
c_mat[1][2] = np.mean(test2_scores[2])
c_mat[2][2] = np.mean(test3_scores[2])

print(c_mat)
# TODO Record all the classification scores and generate a confusion matrix
#-------------------------------------- END SECTION --------------------------------------#

# %% Perform validation on a probe sample (not in the database) and match it against the trained model

# TODO Record the classification score

#-------------------------------------- END SECTION --------------------------------------#

# %% Evaluation

# TODO Evaluate the error rates using the confusion matrix

#-------------------------------------- END SECTION --------------------------------------#
