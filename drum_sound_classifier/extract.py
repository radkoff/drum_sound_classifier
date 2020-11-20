import collections
import logging
import time
import traceback
import pandas
import inspect
import pickle
import numpy as np
from multiprocessing import get_context
from tqdm import tqdm
from pathlib import Path

from drum_sound_classifier import read_audio, track_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Some of the functionality here is overkill, but it comes from a larger to-be-released library

def _apply_clip_boundaries(time_series, start_seconds, end_seconds, sr):
    if time_series is None:
        return None
    is_mono = len(time_series.shape) == 1
    start_index = int(start_seconds * sr)
    end_index = int(end_seconds * sr) if (end_seconds is not None and not np.isnan(end_seconds)) else None
    return time_series[start_index : end_index] if is_mono else time_series[:,start_index : end_index]

# Returns a dict of {clip index: raw audio}
def _load_audio_for_transforms(audio_path, clips_df, input_types, sr):
    clip_audios = dict()
    if 'raw_audio' in input_types:
        for index, clip in clips_df.iterrows():
            clip_audios[index] = read_audio.load_clip_audio(clip, sr)

    return clip_audios

def _should_skip_serialized(clips_df, input_type, key):
    return all([track_store.has_feature(clip, key) for i, clip in clips_df.iterrows()])

# 'transforms' is a dictionary of { storage_key : (lambda, args) }
# Returns dict of {clip_index: {key: result}}
def _transform_clips_for_track(clips_df, transforms, track_debug_str='', sr=22050, skip_serialized=False):
    assert len(clips_df) > 0 and len(clips_df.audio_path.unique()) == 1

    skipable_keys = []
    input_types = dict()
    for key, (func, args) in transforms.items():
        input_type = list(inspect.signature(func).parameters.keys())[0]
        if skip_serialized and _should_skip_serialized(clips_df, input_type, key):
            skipable_keys.append(key)
        else:
            input_types[key] = input_type

    if skipable_keys:
        logger.info(f'Skipping keys {skipable_keys}')

    audio_path = clips_df.iloc[0].audio_path
    try:
        clip_audios = _load_audio_for_transforms(audio_path, clips_df, input_types.values(), sr)
    except FileNotFoundError:
        logger.error(f'File not found {audio_path}')
        return dict()

    # results is a dict from {clip index -> {key -> result}}
    results = dict(zip(clips_df.index, [dict() for i in range(len(clips_df))]))
    for key, (func, args) in transforms.items():
        if key in skipable_keys:
            continue
        for clip_index, clip in clips_df.iterrows():
            result = None
            try:
                if input_types[key] == 'store':
                    with track_store.load_track_store(clip.store_path, 'r') as store:
                        region_features = track_store.load_clip_group_for_store(store, clip)
                        try:
                            result = func(region_features, **args)
                        except KeyError:
                            logger.warning('{} was missing features and could not complete'.format(clip.store_path))
                else:
                    result = func(clip_audios[clip_index] if clip_index in clip_audios else None, **args)
            except Exception as e:
                logger.warning(f'Unable to apply lambda {key} to '
                               f'{track_debug_str} ({clip.start_time}, {clip.end_time})')
                logger.warning(traceback.format_exc())

            # If the extractor passes back multiple results in a dict-like object, save under 'key_{resultkey}'
            if isinstance(result, collections.Mapping):
                for return_key, return_result in result.items():
                    results[clip_index][f'{key}_{return_key}'] = return_result
            else:
                results[clip_index][key] = result
    return results


def _write_results_to_df(clips_df, all_results):
    clips_df = clips_df.copy()
    for clip_index, results in all_results.items():
        for key, result in results.items():
            if result is not None and\
                    (key not in clips_df.loc[clip_index] or pandas.isnull(clips_df.loc[clip_index][key])):
                if isinstance(result, np.ndarray):
                    result = result.tolist()
                # We can store lists in DataFrames but it has to be object dtype first
                if isinstance(result, list) and key not in clips_df:
                    clips_df[key] = pandas.Series(dtype='object')
                clips_df.at[clip_index, key] = result
    return clips_df

def _write_results_to_hdf(clips_df, all_results):
    for clip_index, results in all_results.items():
        successful_keys = [key for (key, value) in results.items() if value is not None]
        if len(successful_keys) > 0:
            logger.debug(f'Setting keys {successful_keys} in {clips_df.loc[clip_index].store_path}')
        for key in successful_keys:
            track_store.set_feature(clips_df.loc[clip_index], key, results[key])
    return clips_df


# Applies a lambda to each of the clips in a dataframe.
# The data fed to the depends on the name of the first arg of the feature:
#   'raw_audio' - Use librosa to read it into a time series
#   'store' - Use the HDF-serialized interim data (Mel coefficients, CQT, etc)
#   'name' - audio file name
#   'metadata' - metadata about the track (pandas Series or dict-like object with 'audio_path', 'store_path', etc)
#   'clip' - row of the clips df
def etl_clips(clips_df,
              clip_lambda,
              key,
              kwargs=dict(),
              serialize=False,
              sr=22050,
              n_jobs=4):
    return etl_clips_batch(clips_df, {key: (clip_lambda, kwargs)}, serialize, sr, n_jobs)

'''
Great documentation goes here
'transforms' is a dictionary of { storage_key : (lambda, args) }
If serialize is True, results are serialized in the HDF store. If false, results are added as attributes of the
dataframe and returned.
parallelize sadly doesn't work from jupyter notebooks
'''
def etl_clips_batch(clips_df, transforms, serialize=False, sr=22050, n_jobs=4):
    new_clips_df = clips_df.copy()
    write_func = _write_results_to_hdf if serialize else _write_results_to_df

    logger.info(f'Running extract jobs {list(transforms.keys())} over {len(new_clips_df)} clips from'
                f' {len(new_clips_df.audio_path.unique())} tracks')
    start = time.time()

    # It's much faster to load tracks once, instead of multiple times for different clips, so launch a subprocess for
    # each unique audio track
    with get_context('spawn').Pool(processes=n_jobs) as pool:
        track_futures = dict()
        for path in new_clips_df.audio_path.unique():
            track_clips = new_clips_df[new_clips_df.audio_path == path]
            args = [track_clips, transforms, path, sr, serialize]
            track_futures[path] = pool.apply_async(_transform_clips_for_track, args=args)
        with tqdm(track_futures.values()) as progress_bar:
            for future in progress_bar:
                # Call get() to ensure each subprocess finishes
                all_results = future.get()
                progress_bar.set_description(Path(new_clips_df.loc[list(all_results.keys())[0]].audio_path).name)
                new_clips_df = write_func(new_clips_df, all_results)

    end = time.time()
    logger.info('Jobs took {:.2f} seconds'.format(end - start))
    return new_clips_df
