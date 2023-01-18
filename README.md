# Deep Learning for Audio and Music Coursework Project

This project contains coursework from the module Deep Learning for Audio and Music at Queen Mary Univerisity of London. The project trains an Autoencoder architecture for a music generation task and a Variational Autoencoder for a music inpainting task. 
The models can be trained separately for separate tasks using the Saarland Music Dataset (https://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html) which consists of over 9.2 hours of piano audio and midi data. For these models, .wav audio files are used from the dataset.

## Data preprocessing

Audio data from the SMD dataset is used for the x-input and the y-prediction, where the x-input data is masked differently for the two separate tasks. During pre-processing, the audio is segmented into 10s clips, at a sample rate of 22050Hz. The audio is then normalised, and converted to mono. Each clip is transformed into a mel-spectrogram of 64 mel bins, with an FFT size of 1024 over a hop length of 512. The mel-spectrogram is also normalised to help the model converge more quickly. 
