from argparse import ArgumentParser
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from drum_sound_classifier import preprocess, extract, drum_descriptors
from drum_sound_classifier.models import inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent

# Preprocessing models for drum descriptor input data
IMPUTATER_PATH = here / 'imputer.pkl'
SCALER_PATH = here / 'scaler.pkl'

MODELS = {
    'lr': LinearRegression(),
    'svc': SVC(),
    'random_forest': RandomForestClassifier(n_estimators=400, min_samples_split=2),
    'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0),
    'knn': KNeighborsClassifier()
}

def train(model_key, train_X, train_y, test_X, test_y, drum_class_labels):
    if model_key == 'all':
        model_keys = MODELS.keys()
    else:
        model_keys = [model_key]

    for key in model_keys:
        model = MODELS[key]
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        logger.info(f"{key}:")
        logger.info(classification_report(test_y, pred, target_names=drum_class_labels, zero_division=0))
        # return model.feature_importances_


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--inputs', type=str, default='cnn_embeddings', choices=['cnn_embeddings', 'descriptors'],
                        help='The source of features on which to build a model')
    parser.add_argument('--model', type=str, default='random_forest', choices=list(MODELS.keys()) + ['all'])
    parser.add_argument('--max_per_class', type=int, default=2000,
                        help='limit common drum types to lessen effects of class imbalance')
    args = parser.parse_args()

    drum_sounds = preprocess.load_dataset()
    drum_sounds = drum_sounds[~drum_sounds.file_drum_type.isna()]
    drum_sounds = drum_descriptors.filter_quiet_outliers(drum_sounds)

    if args.inputs == 'descriptors':
        # Extract hand-crafted drum features
        drum_sounds = extract.etl_clips(drum_sounds, drum_descriptors.low_level_features, '')
    elif args.inputs == 'cnn_embeddings':
        drum_sounds = extract.etl_clips(drum_sounds, inference.embed, 'cnn_embedding')

    # Limit the highest frequency sounds so classes aren't too imbalanced
    drum_sounds = drum_sounds.groupby('file_drum_type').head(args.max_per_class)
    drum_type_labels, unique_labels = pandas.factorize(drum_sounds.file_drum_type)
    drum_sounds = drum_sounds.assign(drum_type_labels=drum_type_labels)
    logger.info(f'Model output can be decoded with the following order of drum types: {list(unique_labels.values)}')
    logger.info(drum_sounds.info())

    # To prevent leakage it's important to use the same train/test split here and cnn
    #TODO serialize test and train separately in preprocess.py to be sure ^
    train_clips_df, val_clips_df = train_test_split(drum_sounds, random_state=0)
    logger.info(f'{len(train_clips_df)} training sounds, {len(val_clips_df)} validation sounds')

    if args.inputs == 'descriptors':
        # The way extract.py / drum_descriptors.py is set up, all descriptor features will start with an underscore
        train_np = train_clips_df.filter(regex='^_', axis=1).to_numpy()
        test_np = val_clips_df.filter(regex='^_', axis=1).to_numpy()

        # There are occassionally random gaps in descriptors, so use imputation to fill in all values
        try:
            imp = pickle.load(open(IMPUTATER_PATH, 'rb'))
        except FileNotFoundError:
            logger.info(f'No cached inputer found, training')
            imp = IterativeImputer(max_iter=25, random_state=0)
            imp.fit(train_np)
            pickle.dump(imp, open(IMPUTATER_PATH, 'wb'))
        train_np = imp.transform(train_np)
        test_np = imp.transform(test_np)
    elif args.inputs == 'cnn_embeddings':
        train_np = np.stack(train_clips_df.cnn_embedding.values)
        test_np = np.stack(val_clips_df.cnn_embedding.values)

    scaler = preprocessing.StandardScaler().fit(train_np)
    train_np = scaler.transform(train_np)
    test_np = scaler.transform(test_np)
    pickle.dump(scaler, open(SCALER_PATH, 'wb'))

    train(args.model, train_np, train_clips_df.drum_type_labels, test_np, val_clips_df.drum_type_labels,
          list(unique_labels.values))

    # Print feature importances
    # importances = train(train_np, train_clips_df.drum_type_labels, test_np, val_clips_df.drum_type_labels)
    # for score, feature in sorted(
    #         zip(importances, train_clips_df.filter(regex='^_', axis=1).drop('_loud_enough_to_analyze', axis=1).columns),
    #         key=operator.itemgetter(0),
    #         reverse=True):
    #     print(f"{feature}\t\t{score}")
