import numpy as np
from torch.utils.data import Dataset

from drum_sound_classifier import track_store

class ClipsDataset(Dataset):
    # clips_df - Dataframe of clips from which to load during training
    # target_feature - column of the DF to learn. Ints for classification, floats for regression
    # mean & std - mean and standard deviation of the dataset, for normalization
    def __init__(self, clips_df, training_data_key, target_feature, mean, std):
        # Make sure all data exists
        if target_feature is not None:
            assert not clips_df[target_feature].hasnans

        # Ensure the df has an index from 0 to len-1
        self.clips_df = clips_df.reset_index(drop=True)
        self.training_data_key = training_data_key
        self.target_feature = target_feature
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.clips_df)

    def __getitem__(self, index):
        clip = self.clips_df.loc[index]

        # Load from disk, normalize
        clip_feature = track_store.load_feature(clip, self.training_data_key)
        clip_feature = np.divide(np.subtract(clip_feature, self.mean), self.std)

        # pytorch expects 3 dimensions (an extra one for channel) so wrap it
        if len(clip_feature.shape) == 2:
            clip_feature = np.expand_dims(clip_feature, 0)

        return clip_feature.astype(np.float32), clip[self.target_feature]
