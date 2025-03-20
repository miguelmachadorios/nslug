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
This script runs the StandardGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid
import os
import warnings
from nslug.algorithms.NSLUG.nslug import NSLUG
from nslug.algorithms.GA.operators.mutators import ga_mutation
from nslug.algorithms.GP.representations.tree_utils import tree_depth
from nslug.config.nslug_config import *
from nslug.selection.selection_algorithms import tournament_selection_max, tournament_selection_min
from nslug.utils.logger import log_settings
from nslug.utils.utils import (get_terminals, validate_inputs, get_best_max, get_best_min)
from nslug.algorithms.GP.operators.mutators import mutate_tree_subtree

def nslug(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
       dataset_name: str = None,
       pop_size: int = nslug_parameters["pop_size"],
       n_iter: int = nslug_solve_parameters["n_iter"],
       p_xo: float = nslug_parameters['p_xo'],
       p_m: float = nslug_parameters['p_m'],
       elitism: bool = nslug_solve_parameters["elitism"], n_elites: int = nslug_solve_parameters["n_elites"],
       log_path: str = None, seed: int = nslug_parameters["seed"],
       log_level: int = nslug_solve_parameters["log"],
       verbose: int = nslug_solve_parameters["verbose"],
       minimization: bool = True,
       fitness_function: str = nslug_solve_parameters["ffunction"],
       initializer: str = nslug_parameters["initializer"],
       n_jobs: int = nslug_solve_parameters["n_jobs"],
       tournament_size: int = 2,
       test_elite: bool = nslug_solve_parameters["test_elite"],
       algorithm: str= nslug_solve_parameters["algorithm"],
       pop_split: float= nslug_parameters["pop_split"]
       
       ):

    """
    Main function to execute the StandardGP algorithm on specified datasets

    Parameters
    ----------
    X_train: (torch.Tensor)
        Training input data.
    y_train: (torch.Tensor)
        Training output data.
    X_test: (torch.Tensor), optional
        Testing input data.
    y_test: (torch.Tensor), optional
        Testing output data.
    dataset_name : str, optional
        Dataset name, for logging purposes
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    p_xo : float, optional
        The probability of crossover in the genetic programming algorithm. Must be a number between 0 and 1 (default is 0.8).
    elitism : bool, optional
        Indicate the presence or absence of elitism.
    n_elites : int, optional
        The number of elites.
    max_depth : int, optional
        The maximum depth for the GP trees.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    log_path : str, optional
        The path where is created the log directory where results are saved. Defaults to `os.path.join(os.getcwd(), "log", "gp.csv")`
    seed : int, optional
        Seed for the randomness
    log_level : int, optional
        Level of detail to utilize in logging.
    verbose : int, optional
       Level of detail to include in console output.
    minimization : bool, optional
        If True, the objective is to minimize the fitness function. If False, maximize it (default is True).
    fitness_function : str, optional
        The fitness function used for evaluating individuals (default is from gp_solve_parameters).
    initializer : str, optional
        The strategy for initializing the population (e.g., "grow", "full", "rhh").
    n_jobs : int, optional
        Number of parallel jobs to run (default is 1).
    prob_const : float, optional
        The probability of a constant being chosen rather than a terminal in trees creation (default: 0.2).
    tree_functions : list, optional
        List of allowed functions that can appear in the trees. Check documentation for the available functions.
    tree_constants : list, optional
        List of constants allowed to appear in the trees.
    tournament_size : int, optional
        Tournament size to utilize during selection. Only applicable if using tournament selection. (Default is 2)
    test_elite : bool, optional
        Whether to test the elite individual on the test set after each generation.

    Returns
    -------
    Tree
        Returns the best individual at the last generation.
    """

    # ================================
    #         Input Validation
    # ================================

    # Setting the log_path
    #if log_path is None:
    log_path = os.path.join(os.getcwd(), "log", "nslug.csv")

    # validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                    
    #                 pop_size=pop_size, n_iter=n_iter,
    #                 elitism=elitism, n_elites=n_elites,  log_path=log_path,
    #                 log=log_level, verbose=verbose,
    #                 minimization=minimization, n_jobs=n_jobs, test_elite=test_elite, fitness_function=fitness_function,
    #                 initializer=initializer, tournament_size=tournament_size)


    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"
    
    assert 0 <= p_m <= 1, "p_xo must be a number between 0 and 1"

    if test_elite and (X_test is None or y_test is None):
        warnings.warn("If test_elite is True, a test dataset must be provided. test_elite has been set to False")
        test_elite = False


    if dataset_name is None:
        warnings.warn("No dataset name set. Using default value of dataset_1.")
        dataset_name = "dataset_1"



    # creating a list with the valid available initializers
    valid_initializers = list(nslug_initializer_options)

    # assuring the chosen initializer is valid
    assert initializer.lower() in nslug_initializer_options.keys(), \
        "initializer must be " + f"{', '.join(valid_initializers[:-1])} or {valid_initializers[-1]}" \
            if len(valid_initializers) > 1 else valid_initializers[0]

    # ================================
    #       Parameter Definition
    # ================================

    if not elitism:
        n_elites = 0

    unique_run_id = uuid.uuid1()

    algo = "nslug"

    #   *************** GP_PI_INIT ***************

    TERMINALS = get_terminals(X_train)
    
    nslug_pi_random_init={"TERMINALS" : TERMINALS,"init_pop_size" : pop_size}
 

    #  *************** GP_PARAMETERS ***************

    nslug_parameters["p_xo"] = p_xo
    nslug_parameters["p_m"] = p_m
    nslug_parameters["pop_size"] = pop_size
    
    nslug_parameters["mutator"] = mutate_tree_subtree(
        nslug_pi_init['init_depth'],  nslug_pi_random_init["TERMINALS"], nslug_pi_init['CONSTANTS'], nslug_pi_init['FUNCTIONS'],
        p_c=nslug_pi_init['p_c']
    )
    
    #ga_parameters["mutator"] = 
    nslug_parameters["initializer"] = nslug_initializer_options[initializer]

    if minimization:
        nslug_parameters["selector"] = tournament_selection_min(tournament_size)
        nslug_parameters["find_elit_func"] = get_best_min
    else:
        nslug_parameters["selector"] = tournament_selection_max(tournament_size)
        nslug_parameters["find_elit_func"] = get_best_max
    nslug_parameters["seed"] = seed
    nslug_parameters["pop_split"]=pop_split
    #   *************** GP_SOLVE_PARAMETERS ***************

    nslug_solve_parameters['run_info'] = [algo, unique_run_id, dataset_name]
    nslug_solve_parameters["log"] = log_level
    nslug_solve_parameters["verbose"] = verbose
    nslug_solve_parameters["log_path"] = log_path
    nslug_solve_parameters["elitism"] = elitism
    nslug_solve_parameters["n_elites"] = n_elites
    nslug_solve_parameters["n_iter"] = n_iter
    nslug_solve_parameters['depth_calculator'] = tree_depth(FUNCTIONS=nslug_pi_init['FUNCTIONS'])
    nslug_solve_parameters["ffunction"] = nslug_fitness_function_options[fitness_function]
    nslug_solve_parameters["algorithm"]=algorithm
    nslug_solve_parameters["n_jobs"] = n_jobs
    nslug_solve_parameters["test_elite"] = test_elite

    # ================================
    #       Running the Algorithm
    # ================================

    optimizer = NSLUG(pi_init=nslug_pi_init,pi_init_rand=nslug_pi_random_init, **nslug_parameters)
    
    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **nslug_solve_parameters
        
    )
    
    log_settings(
        path=log_path[:-4] + "nslug_settings.csv",
        settings_dict=[nslug_solve_parameters,
                       nslug_parameters,
                       nslug_pi_init,
                       settings_dict],
        unique_run_id=unique_run_id,
    )

    return optimizer.elite


if __name__ == "__main__":
    from nslug.datasets.data_loader import load_parkinson
    from nslug.utils.utils import train_test_split

    X, y = load_parkinson(X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    final_tree = nslug(X_train=X_train, y_train=y_train,
                    X_test=X_val, y_test=y_val,
                    dataset_name='load_parkinson', pop_size=250, n_iter=50, n_jobs=2,minimization=False,algorithm="split_population",pop_split=0.3)
    
    #print(final_tree.Tree.fitness)
    #print(final_tree.Tree.repr_)
    #print(final_tree.individual.chromossome)
    #print(final_tree.individual.encoder)
    final_tree.Tree.print_tree_representation()
    predictions = final_tree.Tree.predict(X_test)
    print(float(rmse(y_true=y_test, y_pred=predictions)))
