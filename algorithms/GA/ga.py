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
Genetic Programming (GP) module.
"""

import random
import time
import copy
import numpy as np
import torch
from nslug.algorithms.GA.representations.population import Population_ga
from nslug.algorithms.GA.representations.individual import individual
from nslug.utils.diversity import niche_entropy
from nslug.utils.logger import logger
from nslug.utils.utils import verbose_reporter
from nslug.utils.utils import (get_terminals, validate_inputs, get_best_max, get_best_min)
from nslug.algorithms.GA.operators.crossover_operators import single_point_crossover
from nslug.algorithms.GA.operators.mutators import *

class GA:
    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        find_elit_func,
        p_m=0.2,
        p_xo=0.8,
        pop_size=100,
        seed=0,
        settings_dict=None,
    ):
        """
        Initialize the Genetic Algorithm.

        Parameters
        ----------
        pi_init : dict
            Dictionary with all the parameters needed for candidate solutions initialization.
        initializer : Callable
            Function to initialize the population.
        selector : Callable
            Function to select individuals.
        mutator : Callable
            Function to mutate individuals.
        crossover : Callable
            Function to perform crossover between individuals.
        find_elit_func : Callable
            Function to find elite individuals.
        p_m : float, optional
            Probability of mutation. Default is 0.2.
        p_xo : float, optional
            Probability of crossover. Default is 0.8.
        pop_size : int, optional
            Size of the population. Default is 100.
        seed : int, optional
            Seed for random number generation. Default is 0.
        settings_dict : dict, optional
            Additional settings dictionary.
        """
        self.pi_init = pi_init
        self.selector = selector
        self.p_m = p_m
        self.crossover = single_point_crossover
        self.mutator = ga_mutation
        self.p_xo = p_xo
        self.initializer = initializer
        self.terminals=pi_init["TERMINALS"]
        self.pop_size = pop_size
        self.seed = seed
        self.find_elit_func = find_elit_func
        self.settings_dict = settings_dict

        individual.TERMINALS = pi_init["TERMINALS"]

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        run_info=None,
        max_depth=None,
        ffunction=None,
        n_elites=1,
        depth_calculator=None,
        n_jobs = 1
    ):
        """
        Execute the Genetic Programming algorithm.

        Parameters
        ----------
        X_train : torch.Tensor
            Training data features.
        X_test : torch.Tensor
            Test data features.
        y_train : torch.Tensor
            Training data labels.
        y_test : torch.Tensor
            Test data labels.
        curr_dataset : str
            Current dataset name.
        n_iter : int, optional
            Number of iterations. Default is 20.
        elitism : bool, optional
            Whether to use elitism. Default is True.
        log : int, optional
            Logging level. Default is 0.
        verbose : int, optional
            Verbosity level. Default is 0.
        test_elite : bool, optional
            Whether to evaluate elite individuals on test data. Default is False.
        log_path : str, optional
            Path to save logs. Default is None.
        run_info : list, optional
            Information about the current run. Default is None.
        max_depth : int, optional
            Maximum depth of the tree. Default is None.
        ffunction : function, optional
            Fitness function. Default is None.
        n_elites : int, optional
            Number of elites. Default is 1.
        depth_calculator : function, optional
            Function to calculate tree depth. Default is None.
        n_jobs : int, optional
            The number of jobs for parallel processing. Default is 1.
        """
        # setting the seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        start = time.time()

        # Initialize the population
        population = Population_ga(
            [individual(vec) for vec in self.initializer(**self.pi_init)]
        )

        # evaluating the intial population
#-> aqui vais correr o GP
        population.evaluate(ffunction, X=X_train, y=y_train, n_jobs=n_jobs, caller_id= run_info[1])

        end = time.time()

        # getting the elite(s) from the initial population
        self.elites, self.elite = self.find_elit_func(population, n_elites,"slug")

        # testing the elite on testing data, if applicable
        if test_elite:
            self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True,caller_id= run_info[1])

        # logging the results if the log level is not 0
        log=0
        if log != 0:
            self.log_generation(
                    0, population, end - start, log, log_path, run_info
                )

        # displaying the results on console if verbose level is not 0
        verbose=0
        if verbose != 0:
            verbose_reporter(
                curr_dataset.split("load_")[-1],
                0,
                self.elite.fitness,
                self.elite.test_fitness,
                end - start,
                self.elite.node_count,
            )

        # EVOLUTIONARY PROCESS
        for it in range(1, n_iter + 1):
            # getting the offspring population
            offs_pop, start = self.evolve_population(
                population,
                ffunction,
                elitism,
                X_train,
                y_train,
                n_jobs=n_jobs,
                caller_id=run_info[1]
            )
            # replacing the population with the offspring population (P = P')
            population = offs_pop

            end = time.time()

            # getting the new elite(s)
            self.elites, self.elite = self.find_elit_func(population, n_elites,"slug")

            # testing the elite on testing data, if applicable
            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

            # logging the results if log != 0
            log=0
            if log != 0:
                self.log_generation(
                    it, population, end - start, log, log_path, run_info
                )

            # displaying the results on console if verbose != 0
            if verbose != 0:
                verbose_reporter(
                    run_info[-1],
                    it,
                    self.elite.fitness,
                    self.elite.test_fitness,
                    end - start,
                    self.elite.node_count,
                )

    def evolve_population(
        self,
        population,
        ffunction,
        elitism,
        X_train,
        y_train,
        n_jobs=1,
        caller_id=None
    ):
        """
        Evolve the population for one iteration (generation).

        Parameters
        ----------
        population : Population
            The current population of individuals to evolve.
        ffunction : function
            Fitness function used to evaluate individuals.
        max_depth : int
            Maximum allowable depth for trees in the population.
        depth_calculator : Callable
            Function used to calculate the depth of trees.
        elitism : bool
            Whether to use elitism, i.e., preserving the best individuals across generations.
        X_train : torch.Tensor
            Input training data features.
        y_train : torch.Tensor
            Target values for the training data.
        n_jobs : int, optional
            Number of parallel jobs to use with the joblib library (default is 1).

        Returns
        -------
        Population
            The evolved population after one generation.
        float
            The start time of the evolution process.
        """
        # creating an empty offspring population list
        offs_pop = []

        start = time.time()

        # adding the elite(s) to the offspring population
        if elitism:
            offs_pop.extend(self.elites)

        # filling the offspring population
        while len(offs_pop) < self.pop_size:
            p1, p2 = self.selector(population), self.selector(population)
                # make sure that the parents are different
            while p1 == p2:
                p1, p2 = self.selector(population), self.selector(population)
            p1_tmp= copy.deepcopy(p1)
            p2_tmp= copy.deepcopy(p2)    

            # choosing between crossover and mutation
            if random.random() < self.p_xo: # if crossover is selected
                # generate offspring from the chosen parents
                p1_tmp, p2_tmp  = self.crossover(p1_tmp,p2_tmp)

            if random.random() < self.p_m: # if crossover is selected
                # generate offspring from the chosen parents
                p1_tmp=self.mutator(p1_tmp)
                p2_tmp=self.mutator(p2_tmp)

            # adding the offspring as instances of Tree to the offspring population
            
            offs_pop.extend([p1_tmp,p2_tmp])


        # making sure the offspring population is of the same size as the population
        if len(offs_pop) > population.size:
            offs_pop = offs_pop[: population.size]

        # turning the offspring population into an instance of Population
        #offs_pop = Population(offs_pop)
        # evaluating the offspring population
        offs_pop=Population_ga(offs_pop)
        offs_pop.evaluate(ffunction, X=X_train, y=y_train, n_jobs=n_jobs,caller_id=caller_id)

        # retuning the offspring population and the time control variable
        return offs_pop, start

    def log_generation(
        self, generation, population, elapsed_time, log, log_path, run_info
    ):
        """
        Log the results for the current generation.

        Args:
            generation (int): Current generation (iteration) number.
            population (Population): Current population.
            elapsed_time (float): Time taken for the process.
            log (int): Logging level.
            log_path (str): Path to save logs.
            run_info (list): Information about the current run.

        Returns:
            None
        """
        if log == 2:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                log,
            ]
        elif log == 3:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        elif log == 4:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        else:
            add_info = [self.elite.test_fitness, self.elite.node_count, log]

        logger(
            log_path,
            generation,
            self.elite.fitness,
            elapsed_time,
            float(population.nodes_count),
            additional_infos=add_info,
            run_info=run_info,
            seed=self.seed,
        )
