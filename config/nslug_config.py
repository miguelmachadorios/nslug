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
from nslug.algorithms.GP.operators.crossover_operators import crossover_trees,ecrossover_trees
from nslug.initializers.initializers import rhh, grow, full
from nslug.selection.selection_algorithms import tournament_selection_max

from nslug.evaluators.fitness_functions import *
from nslug.utils.utils import protected_div
import torch

# Define functions and constants
# todo use only one dictionary for the parameters of each algorithm

FUNCTIONS = {
    'add': {'function': torch.add, 'arity': 2},
    'subtract': {'function': torch.sub, 'arity': 2},
    'multiply': {'function': torch.mul, 'arity': 2},
    'divide': {'function': protected_div, 'arity': 2}
}

CONSTANTS = {
    'constant_2': lambda _: torch.tensor(2.0),
    'constant_3': lambda _: torch.tensor(3.0),
    'constant_4': lambda _: torch.tensor(4.0),
    'constant_5': lambda _: torch.tensor(5.0),
    'constant__1': lambda _: torch.tensor(-1.0)
}

# Set parameters
settings_dict = {"p_test": 0.2}

# GP solve parameters
nslug_solve_parameters = {
    "log": 5,
    "verbose": 1,
    "test_elite": True,
    "run_info": None,
    "ffunction": "accuracy",
    "n_jobs": 1,
    "max_depth": 17,
    "n_elites": 1,
    "elitism": True,
    "n_iter": 1000,
    "algorithm": "standard"
}

# GP parameters
nslug_parameters = {
    "initializer": "rhh",
    "selector": tournament_selection_max(2),
    "crossover": crossover_trees(FUNCTIONS),
    "ecrossover": ecrossover_trees(FUNCTIONS),
    "settings_dict": settings_dict,
    "p_xo": 0.8,
    "p_m": 0.2,
    "pop_size": 100,
    "seed": 74,
    "pop_split": 0.5
}

nslug_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0,
    "init_depth": 6
}

nslug_fitness_function_options = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "mae_int": mae_int,
    "signed_errors": signed_errors,
    "accuracy": accuracy
}

nslug_initializer_options = {
    "rhh": rhh,
    "grow": grow,
    "full": full
}