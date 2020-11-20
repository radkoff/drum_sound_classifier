from argparse import ArgumentParser
import logging
from pathlib import Path
import time

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F

from drum_sound_classifier import DRUM_TYPES, preprocess
from drum_sound_classifier.models.clips_dataset import ClipsDataset
from drum_sound_classifier.models import train_nn_supervised, data_loader, CNN_INPUT_SIZE

logger = logging.getLogger(__name__)
here = Path(__file__).parent

# Used for normalization of mel spectrogram data
DATASET_MEAN = -22.72945807
DATASET_STD = 13.65106709

class ConvNet(nn.Module):
    # Designed to work with the input size specified in __init__.py's CNN_INPUT_SIZE
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(12, 4), stride=2)
        self.conv1_batch = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(4, 4), stride=(1, 2))
        self.conv2_batch = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, len(DRUM_TYPES))  # x is a tensor object of size 1x128x259

    # `tensor` is size 1x128x259
    def forward(self, tensor, softmax=True):
        tensor = F.leaky_relu(
            F.max_pool2d(
                self.conv1_batch(self.conv1(tensor)),
                (4, 4)
            )
        )
        logger.debug(tensor.shape)
        tensor = F.leaky_relu(
            F.max_pool2d(
                self.conv2_batch(self.conv2(tensor)),
                (4, 8)
            )
        )
        logger.debug(tensor.shape)

        # Done with convolutions; two fully connected layers, and a softmax
        assert np.prod(tensor.shape[1:]) == 512
        tensor = tensor.view(-1, 512)
        tensor = F.leaky_relu(self.fc1(tensor))
        tensor = F.dropout(tensor, training=self.training)
        tensor = self.fc2(tensor)

        return F.log_softmax(tensor, dim=-1) if softmax else tensor

    # Takes a numpy 128x259 mel spectrogram (in db) as input
    # First normalizes, then does inference and returns an array of class probabilities
    def infer(self, x, softmax=True):
        while len(x.shape) < 4:
            x = np.expand_dims(x, 0)
        x = np.divide(np.subtract(x, DATASET_MEAN), DATASET_STD)
        output = self.forward(torch.from_numpy(x.astype(np.float32)), softmax=softmax)
        softmax_probs = np.exp(output.detach().numpy())[0]     # Model outputs the log probability
        return softmax_probs

    def embed(self, x):
        while len(x.shape) < 4:
            x = np.expand_dims(x, 0)
        x = np.divide(np.subtract(x, DATASET_MEAN), DATASET_STD)
        x = torch.from_numpy(x.astype(np.float32))
        x = F.leaky_relu(F.max_pool2d(self.conv1_batch(self.conv1(x)), (4, 4)))
        x = F.leaky_relu(F.max_pool2d(self.conv2_batch(self.conv2(x)), (4, 8)))
        assert np.prod(x.shape[1:]) == 512
        x = x.view(-1, 512)
        return F.leaky_relu(self.fc1(x)).detach().numpy()[0]


def _log_start(args, experiment_name):
    handler = logging.FileHandler(here / f'logs/training_{experiment_name}.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)-15s %(message)s'))
    logger.addHandler(handler)
    train_nn_supervised.set_handler(handler)
    logger.info('-' * 50)
    logger.info(f'Starting experiment {experiment_name}')
    logger.info(args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='input batch size for validation')
    parser.add_argument('--max_epochs', type=int, default=300, help='max number of epochs to train')
    parser.add_argument('--early_stopping', type=int, default=10, help='how many epochs to go with no improvement in loss before stopping')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.6, help='SGD momentum')
    parser.add_argument('--max_per_class', type=int, default=2000, help='limit common drum types to avoid class imbalance')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--continue_name', type=str, help='allows you to continue previous training, given an experiment name. Look for a model_latest_[experiment].pt and training_[experiment].log to get the name. For now, epoch seconds is used')
    parser.add_argument('--eval', action='store_true', help='Dont train, just load the best model (must provide --continue_name) and print the accuracy')

    args = parser.parse_args()
    experiment_name = args.continue_name if args.continue_name is not None else str(int(time.time()))

    _log_start(args, experiment_name)

    # Assumes preprocess.py has been run, 'file_drum_type' comes from the name of the files
    drum_sounds = preprocess.load_dataset()
    drum_sounds = drum_sounds[~drum_sounds.file_drum_type.isna()]

    # Limit the highest frequency sounds so classes aren't too imbalanced
    drum_sounds = drum_sounds.groupby('file_drum_type').head(args.max_per_class)
    drum_type_labels, unique_labels = pandas.factorize(drum_sounds.file_drum_type)
    drum_sounds = drum_sounds.assign(drum_type_labels=drum_type_labels)
    logger.info(f'Softmax output can be decoded with the following order of drum types: {list(unique_labels.values)}')

    logger.info(f'Normalizing with mean {DATASET_MEAN} and std {DATASET_STD}')
    train_clips_df, val_clips_df = train_test_split(drum_sounds, random_state=0)
    train_dataset = ClipsDataset(train_clips_df, 'mel_spec_model_input', 'drum_type_labels', DATASET_MEAN, DATASET_STD)
    val_dataset = ClipsDataset(val_clips_df, 'mel_spec_model_input', 'drum_type_labels', DATASET_MEAN, DATASET_STD)
    train_loader = data_loader.load(train_dataset, batch_size=args.batch_size, is_train=True,
                                    desired_len=CNN_INPUT_SIZE[1])
    val_loader = data_loader.load(val_dataset, batch_size=args.val_batch_size, is_train=False,
                                  desired_len=CNN_INPUT_SIZE[1])
    logger.info(f'{len(train_clips_df)} training sounds, {len(val_clips_df)} validation sounds')
    logger.info(drum_sounds.file_drum_type.value_counts())

    net = ConvNet()
    logger.info(net)
    if not args.eval:
        train_nn_supervised.run_classification(net, train_loader, val_loader, args.max_epochs, args.early_stopping,
                                               args.lr, args.momentum, args.log_interval, experiment_name,
                                               continueing=args.continue_name is not None)

