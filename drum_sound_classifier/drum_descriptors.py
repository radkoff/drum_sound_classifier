from typing import *

import librosa
import numpy as np

from drum_sound_classifier.read_audio import load_clip_audio, DEFAULT_SR

MAX_FRAMES = 44     # About 1s of audio max, given librosa's hop_length default
MAX_RMS_CUTOFF = 0.02   # If there is no frame with RMS >= MAX_RMS_CUTOFF within MAX_FRAMES, we'll filter it out

SUMMARY_OPS = {
    'avg': np.mean, 'max': np.max, 'min': np.min, 'std': np.std,
    # zero crossing rate
    'zcr': (lambda arr: len(np.where(np.diff(np.sign(arr)))[0]) / float(len(arr)))
}

def filter_quiet_outliers(drum_df):
    # Return a copy of drum_df without sounds that are too quiet for a stable analysis (all RMS frames < 0.02)
    def loud_enough(clip):
        raw_audio = load_clip_audio(clip)
        frame_length = min(2048, len(raw_audio))
        # Use stft for rms input instead of raw audio, like below for consistency
        S, _ = librosa.magphase(librosa.stft(y=raw_audio, n_fft=frame_length))
        rms = librosa.feature.rms(S=S, frame_length=frame_length, hop_length=frame_length//4)

        return max(rms[0][:MAX_FRAMES]) >= MAX_RMS_CUTOFF

    return drum_df[drum_df.apply(loud_enough, axis=1)]

# Works with extract.py if '' is used as the key
def low_level_features(raw_audio):
    features: Dict[str, float] = dict()

    # Edge case: some sounds are so short there isn't even one frames worth of samples
    frame_length = min(2048, len(raw_audio))
    S, _ = librosa.magphase(librosa.stft(y=raw_audio, n_fft=frame_length))
    rms = librosa.feature.rms(S=S, frame_length=frame_length, hop_length=frame_length//4)[0]

    # First get some MPEG-7 standard features
    features = {**features, **_mpeg7(rms)}

    # For the remainder of features, only focus on frames within 1 second
    rms = rms[:MAX_FRAMES]

    # If the signal is too quiet, spectral/mfcc features might not be accurate in places (also, 0.0 will
    # yield nans after log) Discard quiet frames from here on out
    valid_frames = rms >= MAX_RMS_CUTOFF
    rms = rms[valid_frames]

    assert sum(valid_frames) > 0, 'sound too quiet for analysis, filter out using filter_quiet_outliers()'

    # Instead of frame-wise crest factor, just take peak / avg rms
    features['crest_factor'] = rms.max() / rms.mean()

    log_rms = np.log10(rms)
    for op in ['avg', 'std', 'max']:
        features[f'log_rms_{op}'] = SUMMARY_OPS[op](log_rms)

    # We can look at the change in RMS energy between frames (but only if we have >1 frames)
    long_enough_for_gradient = len(log_rms) > 1
    if long_enough_for_gradient:
        log_rms_d = np.gradient(log_rms)
    for op in ['avg', 'std', 'zcr']:
        features[f'log_rms_d_{op}'] = SUMMARY_OPS[op](log_rms_d) if long_enough_for_gradient else np.NaN

    zcr = librosa.feature.zero_crossing_rate(raw_audio, frame_length=frame_length, hop_length=frame_length//4)
    zcr = zcr[0][:MAX_FRAMES][valid_frames]
    loudest_valid_frame = np.argmax(rms)
    for op in ['avg', 'std']:
        features[f'zcr_{op}'] = SUMMARY_OPS[op](zcr)
    features['zcr_loudest'] = zcr[loudest_valid_frame]

    # Add some spectral features
    features = {**features,
                **_spectral_features(S, frame_length, valid_frames, long_enough_for_gradient, loudest_valid_frame)}

    # Now for mfcc's. Not sure if standardization will work or if I have to preprocess the distribution. Log and sqrt
    # won't work because there are negative values. For now no processing
    features = {
        **features,
        **_mfcc_features(S, valid_frames, long_enough_for_gradient, loudest_valid_frame)
    }

    return features

def _mpeg7(rms):
    features = dict()

    peak = np.argmax(rms)
    # Follow the lead of the paper "Computational Models of Similarity for Drum Samples" and focus on when
    # rms reaches 2% of the maximum
    loud_enough = rms >= 0.02 * rms[peak]
    loud_enough_idx = np.where(loud_enough)[0]
    first_loud_enough = loud_enough_idx[0]
    frame_length = (512.0 / DEFAULT_SR)
    attack_time = peak * frame_length - first_loud_enough * frame_length
    # If the attack is 0, we can't take the log so pretend the attack is half of one frame
    log_attack_time = np.log10(attack_time) if attack_time > 0 else np.log10(frame_length / 2.0)
    features['log_attack_time'] = log_attack_time
    # For temporal centroid, we want mean squared amplitude, which is rms squared.
    # Use the frames from the first time it hits 2% of max volume, to the last time it does that
    power = rms ** 2
    last_loud_enough = loud_enough_idx[-1]
    temp_cent_span = power[first_loud_enough: last_loud_enough + 1]
    temp_cent = np.sum(temp_cent_span * np.linspace(0.0, 1.0, len(temp_cent_span))) / np.sum(temp_cent_span) \
        if np.sum(temp_cent_span) > 0 else np.NaN
    features['temp_cent'] = temp_cent
    features['lat_tc_ratio'] = log_attack_time / temp_cent if temp_cent > 0 else np.NaN
    features['duration'] = frame_length * len(temp_cent_span)
    features['release'] = frame_length * (last_loud_enough - peak)

    return features

def _spectral_features(S, frame_length, valid_frames, long_enough_for_gradient, loudest_valid_frame):
    features = dict()

    log_spec_cent = np.log10(librosa.feature.spectral_centroid(
        S=S, n_fft=frame_length, hop_length=frame_length//4)[0][:MAX_FRAMES][valid_frames])
    for op in ['avg', 'std']:
        features[f'log_spec_cent_{op}'] = SUMMARY_OPS[op](log_spec_cent)
    features['log_spec_cent_loudest'] = log_spec_cent[loudest_valid_frame]

    if long_enough_for_gradient:
        log_spec_cent_d = np.gradient(log_spec_cent)
    for op in ['avg', 'std', 'zcr']:
        features[f'log_spec_cent_d_{op}'] = SUMMARY_OPS[op](log_spec_cent_d) if long_enough_for_gradient else np.NaN

    log_spec_band = np.log10(librosa.feature.spectral_bandwidth(S=S, n_fft=frame_length, hop_length=frame_length//4)
                             [0][:MAX_FRAMES][valid_frames])
    for op in ['avg', 'std', 'max']:
        features[f'log_spec_band_{op}'] = SUMMARY_OPS[op](log_spec_band)
    features['log_spec_band_d_avg'] = np.mean(np.gradient(log_spec_band)) if long_enough_for_gradient else np.NaN

    spec_flat = librosa.feature.spectral_flatness(
        S=S, n_fft=frame_length, hop_length=frame_length//4)[0][:MAX_FRAMES][valid_frames]
    for op in ['avg', 'max', 'min', 'std']:
        features[f'spec_flat_{op}'] = SUMMARY_OPS[op](spec_flat)
    features['spec_flat_loudest'] = spec_flat[loudest_valid_frame]
    features['spec_flat_d_avg'] = np.mean(np.gradient(spec_flat)) if long_enough_for_gradient else np.NaN

    for roll_percent in [.15, .85]:
        spec_rolloff = librosa.feature.spectral_rolloff(
            S=S, roll_percent=roll_percent, n_fft=frame_length, hop_length=frame_length//4
        )[0][:MAX_FRAMES][valid_frames]
        roll_percent_int = int(100 * roll_percent)
        features[f'log_spec_rolloff_{roll_percent_int}_loudest'] = np.log10(spec_rolloff[loudest_valid_frame]) \
            if spec_rolloff[loudest_valid_frame] > 0.0 else np.NaN
        # For some reason some sounds give random 0.0s for the spectral rolloff of certain frames.
        # After log these are -inf and need to be filtered before taking the min
        log_spec_rolloff = np.log10(spec_rolloff[spec_rolloff != 0.0])
        features[f'log_spec_rolloff_{roll_percent_int}_max'] = np.max(log_spec_rolloff) \
            if len(log_spec_rolloff) > 0 else np.NaN
        features[f'log_spec_rolloff_{roll_percent_int}_min'] = np.min(log_spec_rolloff) \
            if len(log_spec_rolloff) > 0 else np.NaN

    return features

def _mfcc_features(S, valid_frames, include_gradients, loudest_valid_frame, n_mfcc=13):
    features = dict()

    # Trim the first mfcc because it's basically volume
    mfccs = librosa.feature.mfcc(S=S, n_mfcc=n_mfcc)[1:, :MAX_FRAMES][:, valid_frames]
    n_mfcc -= 1
    # Compute once because it's faster
    transformed_mfcc = {
        'avg': SUMMARY_OPS['avg'](mfccs, axis=1),
        'loudest': mfccs[:,loudest_valid_frame]
    }
    if include_gradients:
        mfcc_d_avg = np.mean(np.gradient(mfccs, axis=1), axis=1)
    for n in range(n_mfcc):
        # std wasn't found to contribute anything
        for op in ['avg', 'loudest']:
            features[f'mfcc_{n}_{op}'] = transformed_mfcc[op][n]
        features[f'mfcc_{n}_d_avg'] = mfcc_d_avg[n] if include_gradients else np.NaN

    return features
