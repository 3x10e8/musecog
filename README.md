# MuseCog

**MuseCog** is a research project exploring the cognitive mechanisms underlying
music perception. It focuses on modeling the listener's expectations of upcoming notes and rhythms,
and provides a tool for investigating perceptive and affective 
processes in behavioral and neuroimaging experiments.

[![hippo](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExajlhYzg3dHpyaGQybGFrNW0xaGg4MXVzYWMxamdzYWpqcDJnc28zciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ESG8KQI9EcrwtPiukE/giphy.gif)](https://www.youtube.com/watch?v=WTHKQMljzXY)


This repository contains:
* PolyRNN: an LSTM model built to yield time-resolved predictions in polyphonic music.
* PolyTNN: equivalent to PolyRNN, with a transfomer architecture.

More information about this framework can be found in Robert et al. (2024),
please use this as citation in publications using this software.

Robert et al., (2024). [Multi-stream predictions in human auditory cortex during natural music listening](). *bioRxiv, 2024-*

## Setup

In this initial release, the code has been tested on Windows 11 with [Python 3.9.13](https://www.python.org/downloads/release/python-3913/).

### CUDA install

GPU acceleration is necessary to train transformer models, and highly recommanded to
train neural networks in general.
To benefit from GPU acceleration, make sure you have a GPU compatible with CUDA,
and install [CUDA](https://developer.nvidia.com/cuda-downloads).

### Python environment
With Windows Command Prompt:

```bash
#clone repository
git clone https://github.com/pl-robert/musecog
cd musecog

#set up a virtual environment
pip install virtualenv
virtualenv venv
.\venv\Scripts\activate.bat

#install python dependencies
pip install -r requirements.txt

#install pytorch manually for cuda support
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### Other dependencies
The ```make_video()``` function allows you to make a video of the output of a model for a MIDI file.
Using this function requires to install:
* [fluidsynth](https://github.com/FluidSynth/fluidsynth/releases)
* [ffmpeg](https://www.ffmpeg.org/download.html)

Note that the path of the ffmpeg folder will have to be given as argument to the function ```make_video()```. Fluidsynth will be automatically located,
and can be installed anywhere.

## Usage
The ```./train_lstm.ipynb``` and ```./train_transformer.ipynb``` notebooks contain the code to
preprocess a MIDI dataset, train a recurrent or transformer model, visualize the model's output,
and export a set of features from your MIDI stimuli. You can open these files with the command ```jupyter notebook train_lstm.ipynb```.

An small midi dataset is provided in ```./data/midi_dataset_example/```. As neural networks require a large amount of data for
training, we recommand using pre-built MIDI datasets like [MAESTRO V3.0.0](https://magenta.tensorflow.org/datasets/maestro).

Pre-trained models PolyRNN and PolyTNN are provided in ```./versions/```, and can be accessed directly with the ```make_video()```
and ```export_features()``` functions.









