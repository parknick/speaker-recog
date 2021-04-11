# %% General Imports

import numpy as np
import scipy.io.wavfile as wav
import python_speech_features as speech

from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix
from utils import prediction_evaluation

# %% High level training parameters

# Change these values when you add or remove training and testing samples
NUM_SUBJECTS = 6 
NUM_PROBE_SAMPLES = 5 
NUM_TRAINING_SAMPLES_PER_SUBJECT = 1
NUM_TESTING_SAMPLES_PER_SUBJECT = 4

SCORE_THRESHOLD = -46 # Identification score threshold

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
    training_set.append(list(wav.read('./wav-files/subject_%d_train_1.wav' % i)))
    for j in range(2, NUM_TRAINING_SAMPLES_PER_SUBJECT + 1):
        training_sample = list(wav.read('./wav-files/subject_%d_train_%d.wav' % (i, j)))
        training_set[i-1][1] = np.concatenate((training_set[i-1][1], training_sample[1]), axis=0)

# Read in all of the test wav files into the test set
# Each entry in the test set will be a tuple with the folowing elements
#   Sample freq
#   Wav data from each test file for the subject
for i in range(1, NUM_SUBJECTS + 1):
    for j in range(1, NUM_TESTING_SAMPLES_PER_SUBJECT + 1):
        testing_sample = list(wav.read('./wav-files/subject_%d_test_%d.wav' % (i, j)))
        testing_set.append(testing_sample)
        testing_labels.append(i)
#-------------------------------------- END SECTION --------------------------------------#

# %% Perform feature extraction on the data

training_features = []
testing_features = []

# Extracting the melcesptrum coefficients for each subjects set of training and testing data
for i in range(NUM_SUBJECTS):
    training_features.append(speech.mfcc(training_set[i][1], samplerate=8000))

for i in range(NUM_TESTING_SAMPLES_PER_SUBJECT * NUM_SUBJECTS):
    testing_features.append(speech.mfcc(testing_set[i][1], samplerate=8000))
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
testing_labels_copy = testing_labels.copy()
unidentified_count = 0

prediction_index = 0
scores = []
for i in range(len(testing_features)): # Loop through all of the testing features and perform classification using the trained GMMs
    
    for j in range(len(gmm)): # Determine matching scores with the trained GMMs
        scores.append(gmm[j].score(testing_features[i]))
    
    if(max(scores) >= SCORE_THRESHOLD): # Check whether the score meets the identification threshold
        predicted_labels.append(scores.index(max(scores)) + 1)

        if(predicted_labels[prediction_index] == testing_labels[i]): # Predicted label was correct
            print_str = '\033[1;32;1mSubject %d identified correctly as Subject %d with score %.2f'
        else:
            print_str = '\033[1;31;1mSubject %d mis-identified as Subject %d with score %.2f'

        print(print_str % (testing_labels[i], predicted_labels[prediction_index], max(scores)))
        
        prediction_index = prediction_index + 1
    else:
        print('\033[1;31;1mSubject was unable to be identified by the Speaker Recognition System (Subject %d)' % (testing_labels[i]))
        unidentified_count = unidentified_count + 1
        testing_labels_copy.remove(testing_labels[i])

    scores.clear()

confusionMatrix = confusion_matrix(testing_labels_copy, predicted_labels) # Create confusion matrix with the results of the classification
plot_confusion_matrix(cm=confusionMatrix, target_names = [i for i in range(1, NUM_SUBJECTS+1)])
#-------------------------------------- END SECTION --------------------------------------#

# %% Perform validation on bunch of probe samples (not in the database) and match it against the trained model

num_true_rejections = 0
num_false_acceptances = 0

probe_samples = [] # Load in probe samples
for i in range(1, NUM_PROBE_SAMPLES + 1):
    probe_sample = wav.read('./wav-files/probe_%d.wav' % (i))
    probe_samples.append(probe_sample)

probe_features = [] # Extract probe features
for i in range(NUM_PROBE_SAMPLES):
    probe_features.append(speech.mfcc(probe_samples[i][1], samplerate=8000))

for i in range(len(probe_features)): # Loop through all probe features and validate that the system does not identify them
    
    for j in range(len(gmm)): # Determine matching scores with the trained GMMs
        scores.append(gmm[j].score(probe_features[i]))
    
    if(max(scores) >= SCORE_THRESHOLD): # Check whether the score meets the identification threshold
        print("\033[1;31;1mProbe Subject was mistakenly Identified by System as Subject %d" % (scores.index(max(scores))))
        num_false_acceptances = num_false_acceptances + 1
    else:
        print("\033[1;32;1mProbe Subject was NOT Identified by System")    
        num_true_rejections = num_true_rejections + 1    
    
    scores.clear()
#-------------------------------------- END SECTION --------------------------------------#

# %% Evaluation

prediction_evaluation(predicted_labels, testing_labels_copy, unidentified_count, num_true_rejections, num_false_acceptances) # Determine the prediction metrics
#-------------------------------------- END SECTION --------------------------------------#
