# TODO

from util import *
from model import *

evaluate_model = AttRNNSpeechModel(4, dropout_rate=0.1)
evaluate_model.load_weights('gdrive/My Drive/four_vtb_classifier_model.h5')
