# speaker-recog

> Lab project for ENCM 509 at the University of Calgary

The SpeakerRecognition.py file contains code that utilizes Gaussian Mixture Models and Log-likelihood scores to 
attempt recognition of multiple speakers in a provided 'database'.

The code has been developed in such a way that by changing a couple of high level parameters, the functionality
and corresponding output of the code can be changed depending on the number of training and testing
samples desired.

**ALL AUDIO FILES MUST BE SAVED IN A FOLDER NAMED "wav-files" IN ORDER FOR THIS PROJECT TO WORK.**
The naming scheme for audio files is summarized in the list below:
    
    - Subject files for training: "subject_x_train_y.wav"
    - Subject files for testing: "subject_x_test_y.wav" 
    - Probe files for validation: "probe_z.wav"

Where x is the subject ID, y/z is the corresponding training/testing/probe index.

If any of the high level parameters are changed, the naming of the audio files must be changed to reflect those changes 
if applicable. To ease this operation, 3 sample folders have provided where each folder contains the same files named in a
way that accomodates these changes. To utilize one of the different folders, simply rename it to "wav-files" and rename the
previously used folder to a name that reflects its contents.

Beyond the items mentioned above, the project should be as easy to use as just running the code provided either in a terminal
or using an IDE.