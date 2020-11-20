import argparse
import logging
from pathlib import Path
import pickle
import os
import shutil
import tempfile
import textwrap

import h5py
import librosa
import numpy as np
import pandas

from drum_sound_classifier import extract, read_audio, drum_descriptors, DRUM_TYPES
from drum_sound_classifier.models import CNN_INPUT_SIZE


logger = logging.getLogger(__name__)
here = Path(__file__).parent

DATAFRAME_FILENAME = 'dataset.pkl'
INTERIM_PATH = here/'../data/interim'


def _trim(raw_audio, sr=read_audio.DEFAULT_SR):
    '''
    Finds the first onset of the sound, returns a good start time and end time that isolates the sound
    :param raw_audio: np array of audio data, from librosa.load
    :param sr: sample rate
    :return: dict with 'start' and 'end', in seconds
    '''
    start = 0.0
    end = None

    # Add an empty second so that the beginning onset is recognized
    silence_to_add = 1.0
    raw_audio = np.append(np.zeros(int(silence_to_add * sr)), raw_audio)

    # Spectral flux
    hop_length = int(librosa.time_to_samples(1. / 200, sr=sr))
    onsets = librosa.onset.onset_detect(y=raw_audio, sr=sr, hop_length=hop_length, units='time')

    if len(onsets) == 0:
        return {'start': start, 'end': end}
    elif len(onsets) > 1:
        # If there are multiple onsets, cut it off just before the second one
        end = onsets[1] - (silence_to_add + 0.01)

    start = max(onsets[0] - (silence_to_add + 0.01), 0.0)
    return {'start': start, 'end': end}

def _extract_cnn_input(raw_audio):
    frame_length = min(2048, len(raw_audio))
    mel_spec = librosa.core.power_to_db(librosa.feature.melspectrogram(
        y=raw_audio, sr=read_audio.DEFAULT_SR, n_fft=frame_length,
        hop_length=frame_length//4, n_mels=CNN_INPUT_SIZE[0])
    )
    # Truncate number of frames stored
    m = min(CNN_INPUT_SIZE[1], mel_spec.shape[1])
    return mel_spec[:, 0:m]

def load_dataset():
    return pickle.load(open(INTERIM_PATH / DATAFRAME_FILENAME, 'rb'))

# Recursively crawls a directory (input_dir_path) looking for audio files, and creates a pandas DataFrame where each
# row is a clip. It also creates a parallel directory structure in 'output_path',
# where each file is an HDF store, which can be used to compute interim features (fourier transform,
# constant-Q transform, beat locations, etc.) for the audio of clips of that track.
# So, if there is an 'input_dirname/foo/Song.mp3', there will be a 'output_path/foo/Song.h5'
# The resulting DataFrame summarizing the entire library of clips is also serialized as a pickle, dataset.pkl
# Rather than interact with these h5 stores directly, use load_dataset() to read data or extract.py to run jobs
def read_drum_library(input_dir_path):
    INTERIM_PATH.mkdir(parents=True, exist_ok=True)
    # Clear any previous data
    shutil.rmtree(INTERIM_PATH)
    INTERIM_PATH.mkdir()

    logger.info(f'Searching for audio files found in {input_dir_path}, setting up HDF stores in {INTERIM_PATH}')

    library_store_path = INTERIM_PATH.joinpath(DATAFRAME_FILENAME)
    dataframe_rows = []
    for input_file in input_dir_path.glob('**/*.*'):
        absolute_path_name = input_file.resolve().as_posix()
        if not read_audio.can_load_audio(absolute_path_name):
            continue

        # Create HDF store for the track
        #TODO are these even necessary? How slow is it to read/compute by dataloader on the fly?
        # If this is needed, do this initialization in track_store.py or lazily in extract.py?
        relative_file_path = input_file.relative_to(input_dir_path)
        file_store_path = INTERIM_PATH.joinpath(relative_file_path).with_suffix('.h5').absolute()
        logger.debug('Opening ' + file_store_path.as_posix())
        file_store_path.parents[0].mkdir(parents=True, exist_ok=True)
        f = h5py.File(file_store_path.as_posix(), mode='a')
        f.close()

        properties = {
            'audio_path': absolute_path_name,
            'store_path': file_store_path.as_posix(),
            'file_stem': Path(absolute_path_name).stem.lower(),
            'start_time': 0.0,
            'end_time': np.NaN
        }
        # Tack on the original file duration (will have to load audio)
        audio = read_audio.load_raw_audio(absolute_path_name, fast=True)
        if audio is None:
            continue  # can_load_audio check above usually catches bad files, but sometimes not
        properties['orig_duration'] = len(audio) / float(read_audio.DEFAULT_SR)

        dataframe_rows.append(properties)

    dataframe = pandas.DataFrame(dataframe_rows)

    pickle.dump(dataframe, open(library_store_path, 'wb'))
    return library_store_path.absolute().as_posix()


def make(args):
    drum_lib_path = Path(os.environ.get('DRUM_LIB_PATH') or args.drum_lib_path)
    clips_path = read_drum_library(drum_lib_path)

    drum_sounds = load_dataset()
    drum_sounds = drum_sounds[drum_sounds.orig_duration < args.max_seconds]
    drum_sounds = drum_sounds.reset_index(drop=True)

    for drum_type_class in DRUM_TYPES:
        drum_sounds.loc[drum_sounds.file_stem.str.contains(drum_type_class), 'file_drum_type'] = drum_type_class
    logger.info(f'After removing those with duration > {args.max_seconds} seconds, there are {len(drum_sounds)} drum sounds.'
                f' Class labels found in {sum(~drum_sounds.file_drum_type.isna())} file names.')

    # Some drum sounds have silence at the beginning, and more than one onset. We only want one onset,
    # and for it to play right away
    drum_sounds = extract.etl_clips(drum_sounds, _trim, 'trimmed')
    if 'trimmed_end' not in drum_sounds:
        drum_sounds['trimmed_end'] = np.NaN  # In case no drum sounds had to be trimmed, NaN means there's no end time
    drum_sounds['start_time'] = drum_sounds['trimmed_start']
    drum_sounds['end_time'] = drum_sounds['trimmed_end']
    drum_sounds = drum_sounds.drop(['trimmed_start', 'trimmed_end'], axis=1)

    # Some hand-crafted features aren't valid for very quiet sounds (rms < 0.02), so filter them out now for valid
    # comparison to other feature representations
    drum_sounds = drum_descriptors.filter_quiet_outliers(drum_sounds)

    if args.extract_spectrograms:
        # Extract mel-spectrogram data, needed for CNN model
        extract.etl_clips(drum_sounds, _extract_cnn_input, 'mel_spec_model_input', serialize=True)

    logger.info(f'Writing clips dataframe of size {len(drum_sounds)} to {clips_path}')
    pickle.dump(drum_sounds, open(clips_path, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(f'''
            This script turns a bunch of drum sound audio flies into a pandas DataFrame. Columns:
                audio_path
                end_time - in seconds
                file_stem
                start_time - in seconds (possibly >0.0 if --trim)
                orig_duration - duration in seconds before any trimming
                file_drum_type - drum class extracted from the filename, if present. One of {DRUM_TYPES}

            It can also extract mel spectrogram data for feeding into NN models
        ''')
    )

    parser.add_argument('--drum_lib_path', type=Path, default=here/'../data/audio',
                        help='Path to drum library; nested directories are fine, and any nonaudio/invalid files will'
                             ' be skipped. If absent, will look in `data/audio/`')
    parser.add_argument('--max_seconds', type=float, default=5.0,
                        help='If a sound is longer than this duration, exclude it')
    parser.add_argument('--trim', action='store_true', dest='trim',
                        help='Ignore silence (or near silence) before the first onset of each audio file. Also, check '
                             'for multiple onsets and ignore anything beyond the first.')
    parser.add_argument('--no_trim', action='store_false', dest='trim')
    parser.add_argument('--extract_spectrograms', action='store_true', dest='extract_spectrograms',
                        help='Extract and serialize spectrogram data (needed if you want to run CNN model)')
    parser.add_argument('--no_extract_spectrograms', action='store_false', dest='extract_spectrograms')
    parser.set_defaults(extract_spectrograms=True)
    parser.set_defaults(trim=True)

    args = parser.parse_args()
    make(args)
