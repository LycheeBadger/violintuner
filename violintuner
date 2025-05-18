import pyaudio
import numpy as np
import aubio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import threading
import time

# Audio configuration
CHUNK = 2048  # Audio buffer size
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100  # Sampling rate
TOLERANCE = 0.8  # Pitch detection confidence threshold
CENT_THRESHOLD = 10  # Cents threshold for correct note (±10 cents)

# Standard A4 frequency (440 Hz)
A4_FREQ = 440.0
A4_MIDI = 69

# MIDI note to note name mapping
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_midi(freq):
    """Convert frequency to MIDI note number."""
    if freq <= 0:
        return None
    return 12 * np.log2(freq / A4_FREQ) + A4_MIDI

def midi_to_note(midi):
    """Convert MIDI note to note name and octave."""
    if midi is None:
        return "None"
    midi = round(midi)
    octave = midi // 12 - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"

def cents_difference(freq, target_freq):
    """Calculate cents difference between detected and target frequency."""
    if freq <= 0 or target_freq <= 0:
        return float('inf')
    return 1200 * np.log2(freq / target_freq)

def get_closest_note(freq):
    """Find the closest musical note to the detected frequency."""
    midi = freq_to_midi(freq)
    if midi is None:
        return None, None
    midi_rounded = round(midi)
    closest_freq = A4_FREQ * (2 ** ((midi_rounded - A4_MIDI) / 12))
    return midi_rounded, closest_freq

# Initialize audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize aubio pitch detection
pitch_o = aubio.pitch("default", CHUNK, CHUNK, RATE)
pitch_o.set_unit("Hz")
pitch_o.set_tolerance(TOLERANCE)

# GUI setup
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 3)
ax.set_axis_off()

# Define blocks
top_block = Rectangle((0, 2), 1, 1, fc='indianred', ec='black')
middle_block = Rectangle((0, 1), 1, 1, fc='forestgreen', ec='black')
bottom_block = Rectangle((0, 0), 1, 1, fc='indianred', ec='black')
ax.add_patch(top_block)
ax.add_patch(middle_block)
ax.add_patch(bottom_block)

# Text for note display
note_text = ax.text(0.5, 1.5, "None", ha='center', va='center', fontsize=20, color='white')

# State variables
current_note = "None"
too_high = False
too_low = False
correct = False
last_update = 0
flash_duration = 0.2  # Flash duration in seconds

def update_colors():
    """Update block colors based on note accuracy."""
    global last_update
    current_time = time.time()
    
    # Default greyish colors when no note is detected
    if current_note == "None":
        top_block.set_facecolor('lightcoral')
        middle_block.set_facecolor('darkseagreen')
        bottom_block.set_facecolor('lightcoral')
        return

    # Flash colors based on accuracy
    if current_time - last_update < flash_duration:
        if too_high:
            top_block.set_facecolor('red')
            middle_block.set_facecolor('darkseagreen')
            bottom_block.set_facecolor('lightcoral')
        elif too_low:
            top_block.set_facecolor('lightcoral')
            middle_block.set_facecolor('darkseagreen')
            bottom_block.set_facecolor('red')
        elif correct:
            top_block.set_facecolor('lightcoral')
            middle_block.set_facecolor('limegreen')
            bottom_block.set_facecolor('lightcoral')
    else:
        # Revert to default colors after flash
        top_block.set_facecolor('indianred')
        middle_block.set_facecolor('forestgreen')
        bottom_block.set_facecolor('indianred')

def audio_processing():
    """Process audio input and update note detection."""
    global current_note, too_high, too_low, correct, last_update
    while True:
        try:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)
            
            # Detect pitch
            pitch = pitch_o(samples)[0]
            confidence = pitch_o.get_confidence()

            if confidence < TOLERANCE or pitch < 20:
                current_note = "None"
                too_high = False
                too_low = False
                correct = False
                continue

            # Find closest note
            midi_rounded, closest_freq = get_closest_note(pitch)
            if midi_rounded is None:
                current_note = "None"
                continue

            current_note = midi_to_note(midi_rounded)
            cents_diff = cents_difference(pitch, closest_freq)

            # Determine if note is too high, too low, or correct
            too_high = cents_diff > CENT_THRESHOLD
            too_low = cents_diff < -CENT_THRESHOLD
            correct = abs(cents_diff) <= CENT_THRESHOLD

            if tooเดียว

System: You are Grok 3 built by xAI.

### Response to User's Request

The provided code is incomplete, as it cuts off at the end of the `audio_processing` function. Below, I’ll complete the program by finishing the `audio_processing` function and adding the necessary logic to update the GUI in real-time. The program will detect violin notes, estimate the closest musical note, and display the results in a three-block interface (top and bottom red, middle green) as specified. The blocks will be greyish when no note is played, the middle block will show the detected note, and the blocks will flash bright red (top for too high, bottom for too low) or bright green (middle for correct) when a note is played.

### Completed Program Code

```python
import pyaudio
import numpy as np
import aubio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import threading
import time

# Audio configuration
CHUNK = 2048  # Audio buffer size
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100  # Sampling rate
TOLERANCE = 0.8  # Pitch detection confidence threshold
CENT_THRESHOLD = 10  # Cents threshold for correct note (±10 cents)

# Standard A4 frequency (440 Hz)
A4_FREQ = 440.0
A4_MIDI = 69

# MIDI note to note name mapping
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_midi(freq):
    """Convert frequency to MIDI note number."""
    if freq <= 0:
        return None
    return 12 * np.log2(freq / A4_FREQ) + A4_MIDI

def midi_to_note(midi):
    """Convert MIDI note to note name and octave."""
    if midi is None:
        return "None"
    midi = round(midi)
    octave = midi // 12 - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"

def cents_difference(freq, target_freq):
    """Calculate cents difference between detected and target frequency."""
    if freq <= 0 or target_freq <= 0:
        return float('inf')
    return 1200 * np.log2(freq / target_freq)

def get_closest_note(freq):
    """Find the closest musical note to the detected frequency."""
    midi = freq_to_midi(freq)
    if midi is None:
        return None, None
    midi_rounded = round(midi)
    closest_freq = A4_FREQ * (2 ** ((midi_rounded - A4_MIDI) / 12))
    return midi_rounded, closest_freq

# Initialize audio stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize aubio pitch detection
pitch_o = aubio.pitch("default", CHUNK, CHUNK, RATE)
pitch_o.set_unit("Hz")
pitch_o.set_tolerance(TOLERANCE)

# GUI setup
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 3)
ax.set_axis_off()

# Define blocks
top_block = Rectangle((0, 2), 1, 1, fc='indianred', ec='black')
middle_block = Rectangle((0, 1), 1, 1, fc='forestgreen', ec='black')
bottom_block = Rectangle((0, 0), 1, 1, fc='indianred', ec='black')
ax.add_patch(top_block)
ax.add_patch(middle_block)
ax.add_patch(bottom_block)

# Text for note display
note_text = ax.text(0.5, 1.5, "None", ha='center', va='center', fontsize=20, color='white')

# State variables
current_note = "None"
too_high = False
too_low = False
correct = False
last_update = 0
flash_duration = 0.2  # Flash duration in seconds

def update_colors():
    """Update block colors based on note accuracy."""
    global last_update
    current_time = time.time()
    
    # Default greyish colors when no note is detected
    if current_note == "None":
        top_block.set_facecolor('lightcoral')
        middle_block.set_facecolor('darkseagreen')
        bottom_block.set_facecolor('lightcoral')
    else:
        # Flash colors based on accuracy
        if current_time - last_update < flash_duration:
            if too_high:
                top_block.set_facecolor('red')
                middle_block.set_facecolor('darkseagreen')
                bottom_block.set_facecolor('lightcoral')
            elif too_low:
                top_block.set_facecolor('lightcoral')
                middle_block.set_facecolor('darkseagreen')
                bottom_block.set_facecolor('red')
            elif correct:
                top_block.set_facecolor('lightcoral')
                middle_block.set_facecolor('limegreen')
                bottom_block.set_facecolor('lightcoral')
        else:
            # Revert to default colors after flash
            top_block.set_facecolor('indianred')
            middle_block.set_facecolor('forestgreen')
            bottom_block.set_facecolor('indianred')
    
    # Update note display
    note_text.set_text(current_note)
    plt.draw()

def audio_processing():
    """Process audio input and update note detection."""
    global current_note, too_high, too_low, correct, last_update
    while True:
        try:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)
            
            # Detect pitch
            pitch = pitch_o(samples)[0]
            confidence = pitch_o.get_confidence()

            if confidence < TOLERANCE or pitch < 20:
                current_note = "None"
                too_high = False
                too_low = False
                correct = False
            else:
                # Find closest note
                midi_rounded, closest_freq = get_closest_note(pitch)
                if midi_rounded is None:
                    current_note = "None"
                    too_high = False
                    too_low = False
                    correct = False
                else:
                    current_note = midi_to_note(midi_rounded)
                    cents_diff = cents_difference(pitch, closest_freq)

                    # Determine if note is too high, too low, or correct
                    too_high = cents_diff > CENT_THRESHOLD
                    too_low = cents_diff < -CENT_THRESHOLD
                    correct = abs(cents_diff) <= CENT_THRESHOLD
                    last_update = time.time()

            # Update GUI in main thread
            plt.pause(0.001)
        except Exception as e:
            print(f"Error in audio processing: {e}")
            break

def gui_update():
    """Continuously update the GUI."""
    while True:
        update_colors()
        plt.pause(0.01)

# Start audio processing in a separate thread
audio_thread = threading.Thread(target=audio_processing, daemon=True)
audio_thread.start()

# Start GUI update in the main thread
plt.show()

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
