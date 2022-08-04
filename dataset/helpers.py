"""
Helper functions to create melodies and music21 score objects
"""
import gzip
import json
import urllib

import numpy as np
from scipy.stats import entropy
from music21 import chord, analysis, note, stream
import music21
from fractions import Fraction
from typing import Union

from utils.helpers import to_numpy

CONTINUE_SYMBOL = '~'

TICK_VALUES = [0, Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]


def rhythmic_entropy(score, base=None):
    durations = []
    for el in score.recurse().getElementsByClass([note.Note, note.Rest, chord.Chord]):
        durations.append(el.duration.quarterLength)
    if base is None:
        base = 16  # for 16 possible classes
    value, counts = np.unique(durations, return_counts=True)
    return entropy(counts, base=base)


# calculate the rhythmic entropy for the entire score, maps it to one of ten values
def calc_rhtyhmic_entropy_category(score, base=None):
    return int(round(rhythmic_entropy(score, base), 1) * 10)


def calc_unique_pitch_number_category(score):
    # alle values sind zwischen 2 und 11
    return get_unique_pitch_number(score) - 2


def get_unique_pitch_number(score):
    pitches = []
    pitches_without_none = []
    for elem in score.recurse().getElementsByClass([note.Note, chord.Chord]):
        if type(elem) is note.Note:
            if elem.pitch not in pitches:
                pitches.append(elem.pitch)
        elif type(elem) is chord.Chord:
            for p in elem.pitches:  # go through all pitches in chord
                if p not in pitches:
                    pitches.append(p)
        pitches_without_none = list(filter(None, pitches))  # because sometimes there is none from nowhere
    return len(pitches_without_none)


def calc_biggest_interval_category(score):
    return get_biggest_interval(score) - 5


def get_biggest_interval(score):
    p = analysis.discrete.Ambitus()
    return p.getSolution(score).semitones


def get_notes(score: music21.stream.Score) -> list:
    """
    Returns the notes from the music21 score object
    Args:
        score: music21 score object
    Returns:
        list, of music21 note objects
    """
    notes = score.parts[0].flat.notesAndRests
    notes = [n for n in notes if not isinstance(n, music21.harmony.ChordSymbol)]
    return notes


def is_score_on_ticks(score: music21.stream.Score, tick_values: list) -> bool:
    """
    Checks if the notes in a score are on ticks
    Args:
        score: music21 score object
        tick_values: list of allowed tick values
    """
    notes = get_notes(score)
    eps = 1e-5
    for n in notes:
        _, d = divmod(n.offset, 1)
        flag = False
        for tick_value in tick_values:
            if tick_value - eps < d < tick_value + eps:
                flag = True
        if not flag:
            return False
    return True


def has_overlapping_notes(score: music21.stream.Score) -> bool:
    notes = get_notes(score)

    current_time = 0
    for n in notes:
        if n.offset < current_time:
            return True
        current_time = n.offset + n.quarterLength
    return False


def standard_name(note_or_rest: Union[music21.note.Note, music21.note.Rest]) -> str:
    """
    Converts music21 note objects to string
    Args:
        note_or_rest: music21 note.Note or note.Rest object

    Returns:
        str,
    """
    if isinstance(note_or_rest, music21.note.Note):
        return note_or_rest.nameWithOctave
    elif isinstance(note_or_rest, music21.note.Rest):
        return note_or_rest.name
    else:
        raise ValueError("Invalid input. Should be a music21.note.Note or music21.note.Rest object ")


def compute_tick_durations(tick_values: list):
    """
    Computes the tick durations
    Args:
        tick_values: list of allowed tick values
    """
    diff = [n - p
            for n, p in zip(tick_values[1:], tick_values[:-1])]
    diff = diff + [1 - tick_values[-1]]
    return diff


def convert_tensor_to_music_21(score_tensor):
    score = to_numpy(score_tensor[0][0])
    print(score)

    score_stream = music21.stream.Score()
    part = music21.stream.Part()

    total_duration = 0
    duration = 0
    pitch = 0
    for index, x in enumerate(score):
        if x != 0:
            pitch = x
        duration += 1
        if index == len(score) - 1 or not score[index+1] == 0:
            new_note = music21.note.Note()
            new_note.quarterLength = duration/4
            new_note.pitch = pitch
            part.insert(total_duration, new_note)
            total_duration += duration
            duration = 0

    score_stream.insert(part)
    return score_stream


def download_file(url):
    print("Lade url: ", url)
    # Download archive
    try:
        # Read the file inside the .gz archive located at url
        with urllib.request.urlopen(url) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()

        decoded = file_content.decode('utf-8')
        serialized_notes = []
        for line in decoded.splitlines():
            serialized_notes.append(json.loads(line))

        print("Geladen.")

        return serialized_notes

    except Exception as e:
        print(e)
        return 1


def convert_serialized_notes_to_midi(json_score):
    input_seq = json_score['input_sequence']

    note_list = input_seq[0]['notes']

    parsed_score = stream.Score()

    parsed_part = stream.Part()

    for nota in note_list:
        if 'quantizedStartStep' not in nota:
            start_time = 0
        else:
            start_time = int(nota['quantizedStartStep'])

        totalTime = int(nota['quantizedEndStep']) - start_time

        parsed_note = note.Note(nota['pitch'])
        parsed_note.quarterLength = totalTime/4

        parsed_part.insert(start_time/4, parsed_note)

    parsed_score.insert(parsed_part)

    return parsed_score




