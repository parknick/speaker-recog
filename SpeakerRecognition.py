# %% General Imports

import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as speech

from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix

# %% High level training parameters

NUM_SUBJECTS = 3 # Change these values when you add or remove training and testing samples
NUM_TRAINING_SAMPLES_PER_SUBJECT = 1 
NUM_TESTING_SAMPLES_PER_SUBJECT = 1

NUM_G_COMPONENTS = 10 # Number of gaussian mixture components

# %% Read in training and testing data and create two data sets

training_set = [] # Lists to contain the required training and testing data

testing_set = []
testing_labels = []

# Read in all of the training wav files into the training set
# Each entry in the training set will be a tuple with the folowing elements
#   Sample freq
#   Concatenated wav data from each training file for the subject
for i in range(1, NUM_SUBJECTS + 1):
    training_set.append(wav.read('./wav-files/subject_%d_train_1.wav' % i))
    for j in range(2, NUM_TRAINING_SAMPLES_PER_SUBJECT + 1):
        training_sample = wav.read('./wav-files/subject_%d_train_%d.wav' % (i, j))
        training_set[i-1][1] = np.concatenate(training_set[i-1][1], training_sample[1])

# Read in all of the test wav files into the test set
# Each entry in the test set will be a tuple with the folowing elements
#   Sample freq
#   Concatenated wav data from each test file for the subject
for i in range(1, NUM_SUBJECTS + 1):
    testing_set.append(wav.read('./wav-files/subject_%d_test_1.wav' % i))
    testing_labels.append(i)
    for j in range(2, NUM_TESTING_SAMPLES_PER_SUBJECT + 1):
        testing_sample = wav.read('./wav-files/subject_%d_test_%d.wav' % (i, j))
        testing_set[i-1][1] = np.concatenate(testing_set[i-1][1], testing_sample[1])
        testing_labels.append(i)

#-------------------------------------- END SECTION --------------------------------------#

# %% Perform feature extraction on the data

training_features = []
testing_features = []

# Extracting the melcesptrum coefficients for each subjects set of training and testing data
for i in range(NUM_SUBJECTS):
    training_features.append(speech.mfcc(training_set[i][1], training_set[i][0]))

for i in range(NUM_SUBJECTS):
    testing_features.append(speech.mfcc(testing_set[i][1], testing_set[i][0]))

#-------------------------------------- END SECTION --------------------------------------#

# %% Train the GMM models using the training samples

# A GMM is trained for each subject using the training features extracted from each subjects training
# wav files
gmm = []
for i in range(NUM_SUBJECTS):
    gmm.append(GaussianMixture(n_components=NUM_G_COMPONENTS).fit(training_features[i]))

#-------------------------------------- END SECTION --------------------------------------#

# %% Perform classification using the testing data
predicted_labels = []

test1_scores = []
for i in range(len(gmm)):
    test1_scores.append(gmm[i].score(testing_features[0]))
predicted_labels.append(test1_scores.index(max(test1_scores)) + 1)

test2_scores = []
for i in range(len(gmm)):
    test2_scores.append(gmm[i].score(testing_features[1]))
predicted_labels.append(test2_scores.index(max(test2_scores)) + 1)

test3_scores = []
for i in range(len(gmm)):
    test3_scores.append(gmm[i].score(testing_features[2]))
predicted_labels.append(test3_scores.index(max(test3_scores)) + 1)


confusionMatrix = confusion_matrix(testing_labels, predicted_labels)
plot_confusion_matrix(cm=confusionMatrix,
                    target_names = [i for i in range(1, NUM_TESTING_SAMPLES_PER_SUBJECT+1)])
# TODO Record all the classification scores and generate a confusion matrix
#-------------------------------------- END SECTION --------------------------------------#

# %% Perform validation on a probe sample (not in the database) and match it against the trained model

# TODO Record the classification score

#-------------------------------------- END SECTION --------------------------------------#

# %% Evaluation

# TODO Evaluate the error rates using the confusion matrix

#-------------------------------------- END SECTION --------------------------------------#
