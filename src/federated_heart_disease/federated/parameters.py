import numpy as np


def get_model_parameters(model):
    return [model.coef_, model.intercept_]


def set_model_parameters(model, parameters):
    model.coef_ = np.array(parameters[0])
    model.intercept_ = np.array(parameters[1])
    model.classes_ = np.array([0, 1, 2, 3, 4])
