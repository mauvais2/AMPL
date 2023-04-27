#!/usr/bin/env python

from test_retrain_dc_models import *
from atomsci.ddm.utils import llnl_utils

# Train and Predict

def test_reg_config_H1_fit_GraphConvModel():
    H1_init()
    train_and_predict('reg_config_H1_fit_GraphConvModel.json', prefix='H1') # crashes during run


if __name__ == '__main__':
    test_reg_config_H1_fit_GraphConvModel() # the same model as graphconv
