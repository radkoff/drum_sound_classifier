import torch
from pathlib import Path
import numpy as np
import scipy

from drum_sound_classifier import DRUM_TYPES
from drum_sound_classifier.models import train_cnn

here = Path(__file__).parent
# Change this filename to your model
CNN_MODEL_PATH = here / 'model_best_1602920274.pt'

def _get_model(model_path=str(CNN_MODEL_PATH)):
    model = train_cnn.ConvNet()
    model.eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def drum_type(store, model=None):
    probs = class_probs(store, model)
    return DRUM_TYPES[np.argmax(probs)]

def class_probs(store, model=None, softmax=True):
    if model is None:
        model = _get_model()
    return model.infer(store['mel_spec_model_input'], softmax=softmax)

# 2nd to last layer embedding
def embed(store, model=None):
    if model is None:
        model = _get_model()
    return model.embed(store['mel_spec_model_input'])
