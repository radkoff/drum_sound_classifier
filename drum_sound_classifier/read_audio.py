import audioop
import audioread
import logging
import os
import warnings

import librosa
import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore', category=UserWarning, module='librosa.core.audio')

DEFAULT_SR = 22050

def can_load_audio(path_string):
    if not os.path.isfile(path_string):
        return False
    try:
        librosa.core.load(path_string, mono=True, res_type='kaiser_fast', duration=.01)
    except (audioread.NoBackendError, audioread.DecodeError, EOFError, FileNotFoundError, ValueError, audioop.error):
        logger.warning(f'Skipping {path_string}, unreadable')
        return False
    return True

def load_raw_audio(path_string, sr=DEFAULT_SR, offset=0, duration=None, fast=False):
    '''
    Mostly pass-through to librosa, but more defensively
    '''
    try:
        time_series, sr = librosa.core.load(path_string, sr=sr, mono=True, offset=offset, duration=duration,
                                            res_type=('kaiser_fast' if fast else 'kaiser_best'))
    except (audioread.NoBackendError, audioread.DecodeError, EOFError, FileNotFoundError, ValueError, audioop.error):
        logger.warning(f'Can\'t read {path_string}')
        return None

    if (duration is None and time_series.shape[0] > 0)\
            or (duration is not None and time_series.shape[0] + 1 >= int(sr * duration)):
        return time_series
    else:
        logger.warning(f'Can\'t load {path_string} due to length, {time_series.shape[0]} {int(sr * duration)} {duration} {sr}')
        return None

def load_clip_audio(clip, sr=DEFAULT_SR):
    '''
    Clip is a row of a dataframe with a start_time, end_time, and audio_path
    '''
    duration = None if clip.end_time is None or np.isnan(clip.end_time) else (clip.end_time - clip.start_time)
    return load_raw_audio(clip.audio_path, sr=sr, offset=clip.start_time, duration=duration)
