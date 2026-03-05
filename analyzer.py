import librosa
import numpy as np
import scipy.ndimage
from collections import Counter

# --------------------------------------------------
# BASIC AUDIO ANALYSIS
# --------------------------------------------------

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "bpm": round(float(tempo), 2),
        "duration": round(duration, 2),
        "sample_rate": sr
    }

# --------------------------------------------------
# SCALE (KEY) DETECTION
# --------------------------------------------------

def identify_scale(chroma):
    total_pitch_intensity = np.sum(chroma, axis=1)

    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
         2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']

    best_offset = 0
    max_corr = -1

    for i in range(12):
        rotated = np.roll(major_profile, i)
        corr = np.corrcoef(total_pitch_intensity, rotated)[0, 1]
        if corr > max_corr:
            max_corr = corr
            best_offset = i

    return f"{notes[best_offset]} Major"

# --------------------------------------------------
# CHROMA EXTRACTION (IMPROVED)
# --------------------------------------------------

def get_chromagram(file_path):
    y, sr = librosa.load(file_path, mono=True)

    # Harmonic-percussive separation
    y_harmonic, _ = librosa.effects.hpss(y)

    # CQT-based chroma (more stable)
    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=512
    )

    # Smooth chroma
    chroma_smoothed = scipy.ndimage.median_filter(
        chroma,
        size=(1, 9)
    )

    return chroma_smoothed, sr, y_harmonic

# --------------------------------------------------
# CHORD TEMPLATES
# --------------------------------------------------

def get_chord_templates():
    return {
        # Major
        'C:maj':  [1,0,0,0,1,0,0,1,0,0,0,0],
        'C#:maj': [0,1,0,0,0,1,0,0,1,0,0,0],
        'D:maj':  [0,0,1,0,0,0,1,0,0,1,0,0],
        'D#:maj': [0,0,0,1,0,0,0,1,0,0,1,0],
        'E:maj':  [0,0,0,0,1,0,0,0,1,0,0,1],
        'F:maj':  [1,0,0,0,0,1,0,0,0,1,0,0],
        'F#:maj': [0,1,0,0,0,0,1,0,0,0,1,0],
        'G:maj':  [0,0,1,0,0,0,0,1,0,0,0,1],
        'G#:maj': [1,0,0,1,0,0,0,0,1,0,0,0],
        'A:maj':  [0,1,0,0,1,0,0,0,0,1,0,0],
        'A#:maj': [0,0,1,0,0,1,0,0,0,0,1,0],
        'B:maj':  [0,0,0,1,0,0,1,0,0,0,0,1],

        # Minor
        'C:min':  [1,0,0,1,0,0,0,1,0,0,0,0],
        'C#:min': [0,1,0,0,1,0,0,0,1,0,0,0],
        'D:min':  [0,0,1,0,0,1,0,0,0,1,0,0],
        'D#:min': [0,0,0,1,0,0,1,0,0,0,1,0],
        'E:min':  [0,0,0,0,1,0,0,1,0,0,0,1],
        'F:min':  [1,0,0,0,0,1,0,0,1,0,0,0],
        'F#:min': [0,1,0,0,0,0,1,0,0,1,0,0],
        'G:min':  [0,0,1,0,0,0,0,1,0,0,1,0],
        'G#:min': [0,0,0,1,0,0,0,0,1,0,0,1],
        'A:min':  [1,0,0,0,1,0,0,0,0,1,0,0],
        'A#:min': [0,1,0,0,0,1,0,0,0,0,1,0],
        'B:min':  [0,0,1,0,0,0,1,0,0,0,0,1],
    }

# --------------------------------------------------
# SCALE CHORDS FOR KEY BIAS
# --------------------------------------------------

def get_major_scale_chords(root):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']

    major_pattern = [0, 2, 4, 5, 7, 9, 11]
    qualities = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']

    root_index = notes.index(root)

    scale_chords = []

    for interval, quality in zip(major_pattern, qualities):
        note = notes[(root_index + interval) % 12]
        scale_chords.append(f"{note}:{quality}")

    return scale_chords

# --------------------------------------------------
# CHORD IDENTIFICATION (WITH KEY BIAS)
# --------------------------------------------------

def identify_chord(chroma_vector, key_root=None):
    templates = get_chord_templates()

    energy = np.linalg.norm(chroma_vector)
    if energy < 0.1:
        return "Unknown"

    chroma_norm = chroma_vector / energy

    best_chord = "Unknown"
    max_similarity = -1

    for chord_name, template in templates.items():
        template = np.array(template, dtype=float)
        template_norm = template / np.linalg.norm(template)

        similarity = np.dot(chroma_norm, template_norm)

        # Soft key bias
        if key_root is not None:
            scale_chords = get_major_scale_chords(key_root)
            if chord_name not in scale_chords:
                similarity *= 0.85

        if similarity > max_similarity:
            max_similarity = similarity
            best_chord = chord_name

    return best_chord

# --------------------------------------------------
# SMOOTHING
# --------------------------------------------------

def smooth_chord_sequence(chords, window_size=3):
    smoothed = []

    for i in range(len(chords)):
        start = max(0, i - window_size // 2)
        end = min(len(chords), i + window_size // 2 + 1)

        window = chords[start:end]
        valid = [c for c in window if c != "Unknown"]

        if valid:
            smoothed.append(Counter(valid).most_common(1)[0][0])
        else:
            smoothed.append("Unknown")

    return smoothed

# --------------------------------------------------
# TIME FORMAT
# --------------------------------------------------

def format_time(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"

# --------------------------------------------------
# BEAT-ALIGNED CHORD RECOGNITION
# --------------------------------------------------

def run_chord_recognition(chroma, sr, y_harmonic):
    tempo, beat_frames = librosa.beat.beat_track(
        y=y_harmonic,
        sr=sr,
        hop_length=512
    )

    if len(beat_frames) < 2:
        return []

    detected_key = identify_scale(chroma)
    key_root = detected_key.split()[0]

    beat_chords = []
    beat_times = []

    for i in range(len(beat_frames) - 1):
        start_frame = beat_frames[i]
        end_frame = beat_frames[i + 1]

        beat_chroma = np.mean(
            chroma[:, start_frame:end_frame],
            axis=1
        )

        chord = identify_chord(beat_chroma, key_root)

        beat_chords.append(chord)
        beat_times.append(
            librosa.frames_to_time(start_frame, sr=sr)
        )

    smoothed_chords = smooth_chord_sequence(beat_chords, window_size=3)

    chord_results = []
    current_chord = smoothed_chords[0]
    start_time = beat_times[0]

    for i in range(1, len(smoothed_chords)):
        if smoothed_chords[i] != current_chord:
            chord_results.append({
                "start": format_time(start_time),
                "end": format_time(beat_times[i]),
                "chord": current_chord
            })
            current_chord = smoothed_chords[i]
            start_time = beat_times[i]

    chord_results.append({
        "start": format_time(start_time),
        "end": format_time(
            librosa.frames_to_time(beat_frames[-1], sr=sr)
        ),
        "chord": current_chord
    })

    return chord_results

def align_chords_with_lyrics(chord_segments, lyrics_text):
    """
    Rough alignment: distributes lyric lines across chord segments.
    """

    lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]
    
    if not lines or not chord_segments:
        return []

    aligned_output = []

    # Spread chords across lyric lines
    chord_index = 0
    total_chords = len(chord_segments)
    total_lines = len(lines)

    for i, line in enumerate(lines):
        # Map line index proportionally to chord index
        mapped_index = int((i / total_lines) * total_chords)
        mapped_index = min(mapped_index, total_chords - 1)

        chord = chord_segments[mapped_index]["chord"]

        aligned_output.append({
            "chord": chord,
            "line": line
        })

    return aligned_output
