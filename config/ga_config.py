# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This script sets up the configuration dictionaries for the execution of the GP algorithm
"""
from nslug.main_gp import gp
from nslug.initializers.initializers import random_init
from nslug.selection.selection_algorithms import tournament_selection_max
from nslug.evaluators.fitness_functions import *

import torch


# Set parameters
settings_dict = {"p_test": 0.2}

# GP solve parameters
ga_solve_parameters = {
    "log": 1,
    "verbose": 1,
    "test_elite": True,
    "run_info": None,
    "ffunction": "gp",
    "n_jobs": 1,
    "max_depth": 17,
    "n_elites": 1,
    "elitism": True,
    "n_iter": 1000
}

# GP parameters
ga_parameters = {
    "initializer": "random_init",
    "selector": tournament_selection_max(2),
    "settings_dict": settings_dict,
    "p_xo": 0.8,
    "p_m": 0.2,
    "pop_size": 100,
    "seed": 74
}

ga_fitness_function_options = {
    "gp": gp
}

ga_pi_init = {

}

ga_initializer_options = {
    "random_init": random_init
}