# %% [markdown]
# # Homework 1: Sine wave generation and binary classification

# %% [markdown]
# ## Part A - Sine Wave Generation

# %% [markdown]
# ### Setup
# To complete this part, install the required Python libraries:

# %%
import numpy as np
from scipy.io import wavfile

import numpy as np
import glob
from mido import MidiFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %%
# (installation process may be different on your system)
# You don't need to use these libraries, so long as you implement the specified functions
# !pip install numpy
# !pip install scipy
# !pip install IPython
# !pip install glob
# !pip install scikit-learn
# !pip install mido

# %% [markdown]
# 1. Write a function that converts a musical note name to its corresponding frequency in Hertz (Hz)
# 
# `note_name_to_frequency()`
# - **Input**: A string `note_name` combining a note (e.g., `'C'`, `'C#'`, `'D'`, `'D#'`, `'E'`, `'F'`, `'F#'`, `'G'`, `'G#'`, `'A'`, `'A#'`, `'B'`) and an octave number (`'0'` to `'10'`)
# - **Output**: A float representing the frequency in Hz
# - **Details**:
#   - Use A4 = 440 Hz as the reference frequency
#   - Frequencies double with each octave increase (e.g., A5 = 880 Hz) and halve with each decrease (e.g., A3 = 220 Hz)
# 
# - **Examples**:
#   - `'A4'` → `440.0`
#   - `'A3'` → `220.0`
#   - `'G#4'` → `415.3047`

# %%
SAMPLE_RATE = 44100 #The sample rate is 44100 Hz, meaning there are 44,100 samples per second.

#each octave represents a doubling or halving of frequency.
# A4 is 440 Hz
# Given a reference note (A4 = 440 Hz), the frequency of any other note can be calculated using: f = 440 * 2^(n/12)
# where n is the number of semitones away from A4.

#Total Semitones=Note Offset+(Octave−4)×12 - Formula

NOTES_TO_SEMITONES = {
    'C': -9, 'C#': -8, 
    'D': -7, 'D#': -6,
    'E': -5,
    'F': -4, 'F#': -3,
    'G': -2, 'G#': -1,
    'A': 0, 'A#': 1,
    'B': 2
}



def note_name_to_frequency(note_name):
    # Q1: Your code goes here
# Each note has a fixed semitone distance from A4
      
    # Extract the note and octave from the note name
    note, octave = note_name[:-1], int(note_name[-1])

    # Calculate the semitone distance from A4
    semitone_distance = NOTES_TO_SEMITONES[note] + (octave - 4) * 12

    # Calculate the frequency using the formula
    frequency = 440 * (2 ** (semitone_distance / 12))
    
    # Return the frequency rounded to 2 decimal places
    frequency = round(frequency, 2)
      
    return frequency

# %%
note_name_to_frequency('A4')  # Should return 440.0
note_name_to_frequency('C4')  # Should return 261.63
note_name_to_frequency('G3')  # Should return 196.0

# %% [markdown]
# 2. Write a function that linearly decreases the amplitude of a given waveform
# 
# `decrease_amplitude()`
# - **Inputs**:
#   - `audio`: A NumPy array representing the audio waveform at a sample rate of 44100 Hz
# - **Output**: A NumPy array representing the audio waveform at a sample rate of 44100 Hz
# - **Details**:
#   - The function must linearly decrease the amplitude of the input audio. The amplitude should start at 1 (full volume) and decrease gradually to 0 (silence) by the end of the sample explain

# %%
def decrease_amplitude(audio):
    # Q2: Your code goes here
    
    fade_audio = np.linspace(1.0, 0.0, len(audio)) 
    # linspace creates an array of evenly spaced numbers over a specified interval.

    faded_audio = audio * fade_audio
    # Multiply the audio signal by the fade_audio array to decrease amplitude

    # faded_audio = np.clip(faded_audio, -1.0, 1.0)  # Ensure values are within [-1, 1]
    # # # Clip the values to be within the range [-1, 1]            

        
    return faded_audio

# %% [markdown]
# 3. Write a function that adds a delay effect to a given audio where the output is a combination of the original audio and a delayed audio
# 
# `add_delay_effects()`  
# - **Inputs**:  
#   - `audio`: A NumPy array representing the audio waveform, sampled at 44,100 Hz
# - **Output**:  
#   - A NumPy array representing the modified audio waveform, sampled at 44,100 Hz
# - **Details**:
#   - The amplitude of the delayed audio should be 30% of the original audio's amplitude
#   - The amplitude of the original audio should be adjusted to 70% of the original audio's amplitude
#   - The output should combine the original audio (with the adjusted amplitude) with a delayed version of itself
#   - The delayed audio should be offset by 0.5 seconds behind the original audio
# 
# - **Examples**:
#   - The provided files (input.wav and output.wav) provide examples of input and output audio

# %%
# #can use these for visualization if you like, though the autograder won't use ipython

# from IPython.display import Audio, display

# print("Example Input Audio:")
# display(Audio(filename = "input.wav", rate=44100))

# print("Example Output Audio:")
# display(Audio(filename = "output.wav", rate=44100))

# %%
# np.array([A, B, C, D, E])  # Original audio

# Apply 70% volume to original
#   # [0.7A, 0.7B, 0.7C, 0.7D, 0.7E]

# Create delayed version (shift right by 2 samples)
# [C, D, E, A, B] (circular shift)

# Zero-pad the beginning to make it a true delay
 # [0, 0, E, A, B]

# [0, 0, 0.3E, 0.3A, 0.3B]

# Combine signals
# [0.7A+0, 0.7B+0, 0.7C+0.3E, 0.7D+0.3A, 0.7E+0.3B]

# %%
# Since the sample rate is 44,100 Hz, a 0.5-second delay corresponds to:
# delay_samples = 0.5 × 44100 = 22050 samples

def add_delay_effects(audio):
    #Q3: Your code goes here

    # Create a delay effect by shifting the audio signal
    delay_seconds = 0.5
    delay_samples = int(delay_seconds * SAMPLE_RATE)

    # Create output array with extended length
    output_length = len(audio) + delay_samples
    output_audio = np.zeros(output_length)

     # Place original audio (70% volume) at beginning
    output_audio[:len(audio)] = audio * 0.7
    
    # Add delayed audio (30% volume) after the delay period
    output_audio[delay_samples:delay_samples + len(audio)] += audio * 0.3
    
    
    return output_audio


# %% [markdown]
# 4. Write a function that concatenates a list of audio arrays sequentially and a function that mixes audio arrays by scaling and summing them, simulating simultaneous playback
# 
# `concatenate_audio()`
# - **Input**:
#   - `list_of_your_audio`: A list of NumPy arrays (e.g., `[audio1, audio2]`), each representing audio at 44100 Hz
# - **Output**: A NumPy array of the concatenated audio
# - **Example**:
#   - If `audio1` is 2 seconds (88200 samples) and `audio2` is 1 second (44100 samples), the output is 3 seconds (132300 samples)
# 
# `mix_audio()`
# - **Inputs**:
#   - `list_of_your_audio`: A list of NumPy arrays (e.g., `[audio1, audio2]`), all with the same length at 44100 Hz.
#   - `amplitudes`: A list of floats (e.g., `[0.2, 0.8]`) matching the length of `list_of_your_audio`
# - **Output**: A NumPy array representing the mixed audio
# - **Example**:
#   - If `audio1` and `audio2` are 2 seconds long, and `amplitudes = [0.2, 0.8]`, the output is `0.2 * audio1 + 0.8 * audio2`

# %%
def concatenate_audio(list_of_your_audio):
    #Q4: Your code goes here
    # Concatenate the audio signals in the list
    concatenated_audio = np.concatenate(list_of_your_audio)
    return concatenated_audio

# %%
def mix_audio(list_of_your_audio, amplitudes):
    #Q4: Your code goes here
    # Mix the audio signals in the list with specified amplitudes
    mixed_audio = np.zeros_like(list_of_your_audio[0]) # why zeros_like? because we want to create an array of zeros with the same shape as the first audio signal
    for audio, amplitude in zip(list_of_your_audio, amplitudes):
        mixed_audio += audio * amplitude

    return mixed_audio


# %% [markdown]
# 5. Modify your solution to Q2 so that your pipeline can generate sawtooth waves by adding harmonics based on the following equation:
# 
#     $\text{sawtooth}(f, t) = \frac{2}{\pi} \sum_{k=1}^{19} \frac{(-1)^{k+1}}{k} \sin(2\pi k f t)$ 
# 
# - **Inputs**:
#   - `frequency`: Fundamental frequency of sawtooth wave
#   - `duration`: A float representing the duration in seconds (e.g., 2.0)
# - **Output**: A NumPy array representing the audio waveform at a sample rate of 44100 Hz

# %% [markdown]
# The sawtooth wave is a type of periodic waveform that rises linearly and then drops sharply, resembling the teeth of a saw. Unlike a pure sine wave, which consists of a single frequency, a sawtooth wave is rich in harmonics, meaning it contains multiple frequencies at integer multiples of the fundamental frequency.
# 
# Harmonics are higher-frequency sound waves that are integer multiples of a fundamental frequency (the main pitch you hear). They give a sound its unique "color" or timbre.
# 
# When you play a musical note (e.g., 440 Hz = A4), the sound is not just a single pure tone.
# 
# It also contains extra frequencies (880 Hz, 1320 Hz, 1760 Hz, etc.), which are called harmonics or overtones.
# 
# These harmonics make a violin sound different from a flute, even if they play the same note.

# %%
"""
Harmonics in a Sawtooth Wave

The sawtooth wave is constructed by summing multiple sine waves (harmonics) together:

1. 1st harmonic (Fundamental frequency) → f (e.g., 440 Hz)
2. 2nd harmonic → 2f (e.g., 880 Hz)
3. 3rd harmonic → 3f (e.g., 1320 Hz)
4. ... and so on.

Each harmonic has:
- A specific amplitude (strength) → 1/k (amplitude decreases as frequency increases)
- A specific phase (timing) → (-1)^(k+1) (alternates between + and -)
"""

# %%
def create_sawtooth_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sawtooth_wave = np.zeros_like(t)
    
    for k in range(1, 20):  # Sum from k=1 to k=19
        harmonic = (-1) ** (k + 1) / k * np.sin(2 * np.pi * k * frequency * t)
        sawtooth_wave += harmonic
    
    sawtooth_wave *= 2 / np.pi
    
    # Normalize the waveform to the range [-1, 1] to prevent clipping
    # max_amplitude = np.max(np.abs(sawtooth_wave))
    # if max_amplitude > 0:
    #     sawtooth_wave /= max_amplitude
    
    return sawtooth_wave

# %% [markdown]
# ## Part B - Binary Classification
# Train a binary classification model using `scikit-learn` to distinguish between piano and drum MIDI files.

# %% [markdown]
# #### Unzip MIDI Files
# Extract the provided MIDI datasets:
# 
# ```bash
# unzip piano.zip
# unzip drums.zip
# ```
# 
# - `./piano`: Contains piano MIDI files (e.g., `0000.mid` to `2154.mid`)
# - `./drums`: Contains drum MIDI files (e.g., `0000.mid` to `2154.mid`)
# - Source: [Tegridy MIDI Dataset] (https://github.com/asigalov61/Tegridy-MIDI-Dataset)
# 
# These folders should be extracted into the same directory as your solution file

# %% [markdown]
# 6. Write functions to compute simple statistics about the files
# 
# ####  `get_stats()`
# 
# - **Inputs**:
#   - `piano_file_paths`: List of piano MIDI file paths`
#   - `drum_file_paths`: List of drum MIDI file paths`
# - **Output**: A dictionary:
#   - `"piano_midi_num"`: Integer, number of piano files
#   - `"drum_midi_num"`: Integer, number of drum files
#   - `"average_piano_beat_num"`: Float, average number of beats in piano files
#   - `"average_drum_beat_num"`: Float, average number of beats in drum files
# - **Details**:
#   - For each file:
#     - Load with `MidiFile(file_path)`
#     - Get `ticks_per_beat` from `mid.ticks_per_beat`
#     - Compute total ticks as the maximum cumulative `msg.time` (delta time) across tracks
#     - Number of beats = (total ticks / ticks_per_beat)
#   - Compute averages, handling empty lists (return 0 if no files)

# %% [markdown]
# A MIDI file (.mid or .midi) is a standardized format for storing musical performance data—not audio. 
# Instead of recording sound waves (like WAV or MP3), it stores:
# 
# - Which notes are played (pitch, duration, velocity).
# - When they are played (timing, tempo).
# - How they are played (instrument, volume, effects).

# %% [markdown]
# ### Key Concepts for MIDI File Analysis
# 
# 1. **Ticks as Time Unit**:
#     - MIDI files use **ticks** as their fundamental time unit.
#     - A tick represents a small fraction of time within a beat.
# 
# 2. **Ticks Per Beat**:
#     - The `ticks_per_beat` parameter defines how many ticks make up one beat (quarter note).
#     - This value is essential for converting ticks into musical time (e.g., beats or seconds).
# 
# 3. **Total Duration**:
#     - The total duration of a MIDI file is determined by the **longest track** in the file.
#     - This ensures that all tracks are accounted for when calculating the file's length.
# 
# 4. **Counting Musical Content**:
#     - Only **note_on** events with a positive velocity are considered as actual musical content.
#     - These events indicate when a note is played, as opposed to other MIDI messages like tempo changes or control signals.
# 
# Total beats = (Total ticks in the file) / (ticks_per_beat)

# %% [markdown]
# For a MIDI file with:
# 
# ticks_per_beat = 480 (standard resolution)
# 
# Longest track sums to 1920 ticks
# 
# Total beats = 1920 / 480 = 4 beats 

# %%
def get_file_lists():
    piano_files = sorted(glob.glob("./piano/*.mid"))
    drum_files = sorted(glob.glob("./drums/*.mid"))
    return piano_files, drum_files

#The number of beats in a MIDI file refers to the total musical duration of the composition,
# measured in quarter-note beats (the basic pulse of the music).
def get_num_beats(file_path):
    # Q6: Your code goes here
    mid = MidiFile(file_path)
    # Might need: mid.tracks, msg.time, mid.ticks_per_beat

    total_ticks = 0 #Represents the longest track duration in ticks
    
    # Iterate through all tracks and messages
    for track in mid.tracks:
        track_time = 0
        for msg in track:
            track_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                # Only count note_on events with velocity > 0
                pass
        
        # Keep track of the longest track's duration
        if track_time > total_ticks:
            total_ticks = track_time
    
    # Convert ticks to beats
    nBeats = total_ticks / mid.ticks_per_beat
    return nBeats

def get_stats(piano_path_list, drum_path_list):
    piano_beat_nums = []
    drum_beat_nums = []
    for file_path in piano_path_list:
        piano_beat_nums.append(get_num_beats(file_path))
        
    for file_path in drum_path_list:
        drum_beat_nums.append(get_num_beats(file_path))
    
    return {"piano_midi_num":len(piano_path_list),
            "drum_midi_num":len(drum_path_list),
            "average_piano_beat_num":np.average(piano_beat_nums),
            "average_drum_beat_num":np.average(drum_beat_nums)}
    

# %% [markdown]
# 

# %% [markdown]
# 7. Implement a few simple feature functions, to compute the lowest and highest MIDI note numbers in a file, and the set of unique notes in a file
# 
# `get_lowest_pitch()` and `get_highest_pitch()`
# functions to find the lowest and highest MIDI note numbers in a file
# 
# - **Input**: `file_path`, a string (e.g., `"./piano/0000.mid"`)
# - **Output**: An integer (0–127) or `None` if no notes exist
# - **Details**:
#   - Use `MidiFile(file_path)` and scan all tracks
#   - Check `msg.type == 'note_on'` and `msg.velocity > 0` for active notes
#   - Return the minimum (`get_lowest_pitch`) or maximum (`get_highest_pitch`) `msg.note`
# 
# `get_unique_pitch_num()`
# a function to count unique MIDI note numbers in a file
# 
# - **Input**: `file_path`, a string
# - **Output**: An integer, the number of unique pitches
# - **Details**:
#   - Collect `msg.note` from all `'note_on'` events with `msg.velocity > 0` into a set
#   - Return the set’s length
# - **Example**: For notes `["C4", "C4", "G4", "G4", "A4", "A4", "G4"]`, output is 3 (unique: C4, G4, A4)

# %%
p, d = get_file_lists()

def get_lowest_pitch(file_path): #If file contains C4 (60), E4 (64), G4 (67), returns 67
    #Q7-1: Your code goes here

    mid = MidiFile(file_path)
    notes = set()
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.add(msg.note)
    
    return min(notes) if notes else 0


def get_highest_pitch(file_path):
    #Q7-2: Your code goes here
    mid = MidiFile(file_path)
    notes = set()
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.add(msg.note)
        
    return max(notes) if notes else 0


def get_unique_pitch_num(file_path):
    #Q7-3: Your code goes here

    mid = MidiFile(file_path)
    notes = set() # unique since set
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.add(msg.note)
        
    return 0 if not notes else len(notes)


# %% [markdown]
# 8. Implement an additional feature extraction function to compute the average MIDI note number in a file
# 
# `get_average_pitch_value()`
# a function to return the average MIDI note number from a file
# 
# - **Input**: `file_path`, a string
# - **Output**: A float, the average value of MIDI notes in the file
# - **Details**:
#   - Collect `msg.note` from all `'note_on'` events with `msg.velocity > 0` into a set
# - **Example**: For notes `[51, 52, 53]`, output is `52`

# %%
def get_average_pitch_value(file_path):
    #Q8: Your code goes here

    mid = MidiFile(file_path)
    notes = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)

    if notes:
        return sum(notes) / len(notes)
    else:
        return None

# %% [markdown]
# 9. Construct your dataset and split it into train and test sets using `scikit-learn` (most of this code is provided). Train your model to classify whether a given file is intended for piano or drums.
# 
# `featureQ9()`
# 
# Returns a feature vector concatenating the four features described above
# 
# - **Input**: `file_path`, a string.
# - **Output**: A vector of four features

# %%
def featureQ9(file_path):
    # Already implemented: this one is a freebie if you got everything above correct!
    return [get_lowest_pitch(file_path),
            get_highest_pitch(file_path),
            get_unique_pitch_num(file_path),
            get_average_pitch_value(file_path)]

# %% [markdown]
# 10. Creatively incorporate additional features into your classifier to make your classification more accurate.  Include comments describing your solution.

# %%
def featureQ10(file_path):
    #Q10: Your code goes here
    # Extract features from the MIDI file
    features = featureQ9(file_path)
    # Add the number of beats
    features.append(get_num_beats(file_path))
    # Add the average pitch value
    features.append(get_average_pitch_value(file_path))

    return features

# %%
# from sklearn.metrics import accuracy_score

# p , d = get_file_lists()
# # Print the number of files in each category        
# print("Number of piano files:", len(p))
# print("Number of drum files:", len(d))


# # Create feature table
# features = []
# labels = []

# # Extract features for piano files (label = 0)
# for file_path in p:
#     features.append(featureQ10(file_path))
#     labels.append(0)

# # Extract features for drum files (label = 1)
# for file_path in d:
#     features.append(featureQ10(file_path))
#     labels.append(1)

# # Convert to NumPy arrays
# features = np.array(features)
# labels = np.array(labels)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# # Print shapes of the datasets
# print("Training features shape:", X_train.shape)
# print("Testing features shape:", X_test.shape)
# print("Training labels shape:", y_train.shape)
# print("Testing labels shape:", y_test.shape)

# # Train a logistic regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)
# # Evaluate the model

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


