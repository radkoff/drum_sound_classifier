import numpy as np
import h5py
import logging

logger = logging.getLogger(__name__)

'''
Each track of audio in a dataset will have a store set up for serialization/memoization. This store, backed on disk
by the HDF5 protocol, has groups for any clips with the name {offset_duration}. For example, this is what the
contents of the h5 file will look like after extracting 15 seconds worth of Mel Spectrogram data starting at 45 seconds:
FILE_CONTENTS {
 group      /
 group      /45.0_15.0
 dataset    /45.0_15.0/mel_spec
 }
}
Calling load_feature(clip, 'mel_spec') where clip has the right store_path, start_time (45), and end_time (60)
will return it. set_feature and has_feature work as you'd expect.  
If you need access to the HDF5 store itself, this is the recommended pattern:
with track_store.load_track_store(clip.store_path) as store:
    group = track_store.load_clip_group_for_store(store, clip)
'''

def _make_region_key(start_time, end_time):
    start_time = float(start_time)
    duration = None if end_time is None or np.isnan(end_time) else (end_time - start_time)
    # Account for floating point errors, ie 4.9999999 becomes 5.0
    if duration is not None:
        duration = np.round(duration, 3)
    return '{}_{}'.format(np.round(start_time, 3), duration)

def _read_region_key(key):
    start, duration = key.split('_')
    start = 0.0 if start == 'None' else float(start)
    return start, None if duration == 'None' else start + float(duration)

def load_track_store(store_location, mode='a'): return h5py.File(store_location, mode=mode)

def load_clip_group_for_store(store, clip):
    return store.require_group(_make_region_key(clip.start_time, clip.end_time))

def has_feature(clip, key, ignore_clip_region=False):
    with h5py.File(clip.store_path, 'r') as store:
        region_key = _make_region_key(0, None) if ignore_clip_region else _make_region_key(clip.start_time, clip.end_time)
        return region_key in store and key in store[region_key]

def has_features(clip, keys):
    return all([has_feature(clip, k) for k in keys])

def load_feature(clip, key, ignore_clip_region=False):
    with h5py.File(clip.store_path, 'r') as store:
        region_key = _make_region_key(0, None) if ignore_clip_region else _make_region_key(clip.start_time, clip.end_time)
        if region_key not in store:
            raise ValueError(f'Feature group does not exist for start time {clip.start_time} end time {clip.end_time}')
        return store.require_group(region_key)[key][()]

def load_track_feature(store_path, key):
    with h5py.File(store_path, 'r') as store:
        region_key = _make_region_key(0, None)
        if region_key not in store:
            raise ValueError(f'Feature group does not exist for start time {0} end time {None}')
        return store.require_group(region_key)[key][()]

def set_feature(clip, key, data):
    if has_feature(clip, key):
        logger.warning(f'{key} exists in {clip.store_path}, skipping')
    else:
        with h5py.File(clip.store_path, 'a') as store:
            load_clip_group_for_store(store, clip)[key] = data
