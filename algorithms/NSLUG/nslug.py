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
from nslug.algorithms.GP.representations.population import Population
from nslug.algorithms.GP.representations.tree import Tree
from nslug.algorithms.NSLUG.representations.nslug_population import nslug_Population
from nslug.algorithms.NSLUG.representations.nslug_individual import nslug_individual

from nslug.utils.diversity import niche_entropy
from nslug.utils.logger import logger
from nslug.utils.utils import verbose_reporter
from nslug.algorithms.GP.representations.tree_utils import create_random_random_tree,used_terminals
from nslug.algorithms.GA.operators.mutators import *
from nslug.algorithms.GA.operators.crossover_operators import single_point_crossover
from nslug.algorithms.GA.operators.mutators import ga_delete,ga_insert,ga_mutation
from nslug.algorithms.GP.operators.mutators import gp_insert
from nslug.initializers.initializers import random_init

class NSLUG:
    def __init__(
        self,
        pi_init,pi_init_rand,
        initializer,
        selector,
        mutator,
        crossover,
        ecrossover,
        find_elit_func,
        p_m=0.2,
        p_xo=0.8,
        pop_size=100,
        seed=0,
        pop_split=0.5,
        algorithm="standard",
        settings_dict=None,
    ):
        """
        Initialize the Genetic Programming algorithm.

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
        self.pi_init_rand=pi_init_rand
        self.selector = selector
        self.p_m = p_m
        self.mutator=mutator
        self.crossover = crossover
        self.ecrossover=ecrossover
        self.mutator = mutator
        self.ga_crossover = single_point_crossover
        self.ga_mutator = ga_mutation
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.seed = seed
        self.find_elit_func = find_elit_func
        self.settings_dict = settings_dict
        self.FUNCTIONS=pi_init["FUNCTIONS"]
        self.CONSTANTS=pi_init["CONSTANTS"]
        self.pop_split=pop_split

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        individual.TERMINALS = pi_init_rand["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]
        #TERMINALS = pi_init["TERMINALS"]
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
        n_jobs = 1,
        algorithm="standard"
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
        vic_keys = list(individual.TERMINALS.keys())
        
        #[print(individual(tree).chromossome) for tree in  random_init(**self.pi_init_rand)]
        population = nslug_Population(
            [nslug_individual(tree, create_random_random_tree(6,self.FUNCTIONS,individual(tree).chromossome,self.CONSTANTS,0)) for tree in random_init(**self.pi_init_rand)]
        )
        
        for slug_ind in population.population:
            slug_ind.individual= ga_delete(slug_ind.individual,used_terminals(slug_ind.Tree.repr_,slug_ind.individual.TERMINALS))
            
 
        # evaluating the intial population
        population.evaluate(ffunction, X=X_train, y=y_train, n_jobs=n_jobs)

        end = time.time()
        
        # getting the elite(s) from the initial population
        self.elites, self.elite = self.find_elit_func(population, n_elites,"GP")
        
        # testing the elite on testing data, if applicable
        if test_elite:
             self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

        # # logging the results if the log level is not 0
       
        if log != 0:
            self.log_generation(
                    0, population, end - start, log, log_path, run_info
                )

        # displaying the results on console if verbose level is not 0
        if verbose != 0:
            verbose_reporter(
                curr_dataset.split("load_")[-1],
                0,
                self.elite.Tree.fitness,
                self.elite.Tree.test_fitness,
                end - start,
                self.elite.Tree.node_count,
            )

        # EVOLUTIONARY PROCESS
        for it in range(1, n_iter + 1):
            # getting the offspring population
            
            offs_pop, start = self.evolve_population(
                population,
                ffunction,
                max_depth,
                depth_calculator,
                elitism,
                X_train,
                y_train,
                algorithm,
                n_jobs=n_jobs
                
            )
            # replacing the population with the offspring population (P = P')

            population = offs_pop

            end = time.time()
            
            # getting the new elite(s)
            self.elites, self.elite = self.find_elit_func(population, n_elites,"GP")
            
   
            # testing the elite on testing data, if applicable
            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)
            
            # logging the results if log != 0
            if log != 0:
                self.log_generation(
                    it, population, end - start, log, log_path, run_info
                )

            #displaying the results on console if verbose != 0
            if verbose != 0:
                verbose_reporter(
                    run_info[-1],
                    it,
                    self.elite.Tree.fitness,
                    self.elite.Tree.test_fitness,
                    end - start,
                    self.elite.Tree.node_count,
                )

    def evolve_population(
        self,
        population,
        ffunction,
        max_depth,
        depth_calculator,
        elitism,
        X_train,
        y_train,
        algorithm,
        n_jobs=1
        
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
       
        # for ind in population.population:
        #      print(ind.individual.chromossome)
        #      print(dict(sorted(used_terminals(ind.Tree.repr_,individual.TERMINALS).items(), key=lambda item: item[1], reverse=False)) )
        #      print(ind.Tree.fitness)
        #      print("------------------------------------------------------------")
        
        if algorithm=="standard":
        
            while len(offs_pop) < self.pop_size:
                # choosing between crossover and mutation
                p1, p2 = self.selector(population,True), self.selector(population,True)
                    # make sure that the parents are different
                while p1 == p2:
                
                    p1, p2 = self.selector(population,True), self.selector(population,True)
                p1_tmp= copy.deepcopy(p1)
                p2_tmp= copy.deepcopy(p2)    
            
                if random.random() < self.p_xo: # if crossover is selected
                
                    off1_ind,off2_ind=single_point_crossover(p1_tmp.individual,p2_tmp.individual)

                    # generate offspring from the chosen parents
                    offs1_tree, offs2_tree = self.ecrossover(
                        X_train,
                        p1_tmp.Tree.repr_,
                        p2_tmp.Tree.repr_,     
                        tree1_n_nodes=p1_tmp.Tree.node_count,
                        tree2_n_nodes=p2_tmp.Tree.node_count,
                        terminal1=off1_ind.chromossome,
                        terminal2=off2_ind.chromossome,
                        TERMINALS=individual.TERMINALS
                    )
           
                    # assuring the offspring do not exceed max_depth
                    if max_depth is not None:
                        while (
                            depth_calculator(Tree(offs1_tree)) > max_depth
                            or depth_calculator(Tree(offs2_tree)) > max_depth
                        ):
                            offs1_tree, offs2_tree = self.ecrossover(
                                X_train,
                                p1_tmp.Tree.repr_,
                                p2_tmp.Tree.repr_,
                                tree1_n_nodes=p1_tmp.Tree.node_count,
                                tree2_n_nodes=p2_tmp.Tree.node_count,
                                terminal1=off1_ind.chromossome,
                                terminal2=off2_ind.chromossome,
                                TERMINALS=individual.TERMINALS
                            )

                    p1_tmp=nslug_individual(off1_ind.encoder,offs1_tree)
                    p2_tmp=nslug_individual(off2_ind.encoder,offs2_tree)
                    # grouping the offspring in a list to be added to the offspring population
   
                if random.random() < self.p_m: # if mutation was chosen
                    # choosing a parent
                
                
                    off1_ind=ga_insert(p1_tmp.individual)
                    off1_tree=gp_insert(p1_tmp.Tree,off1_ind.chromossome)
                
                    off2_ind=ga_insert(p2_tmp.individual)
                    off2_tree=gp_insert(p2_tmp.Tree,off2_ind.chromossome)
                
                    # making sure the offspring does not exceed max_depth
                    if max_depth is not None:
                        while (
                            depth_calculator(off1_tree) > max_depth
                            or depth_calculator(off2_tree) > max_depth
                        ):
                        
                            off1_ind=ga_insert(p1_tmp.individual)
                            off1_tree=gp_insert(p1_tmp.Tree,off1_ind.chromossome)
                            off2_ind=ga_insert(p2_tmp.individual)
                            off2_tree=gp_insert(p2_tmp.Tree,off2_ind.chromossome)
                        
                    # adding the offspring to a list, to be added to the offspring population
                    p1_tmp=nslug_individual(off1_ind.encoder,off1_tree.repr_)
                    p2_tmp=nslug_individual(off2_ind.encoder,off2_tree.repr_)

                
                # adding the offspring as instances of Tree to the offspring population
                offs_pop.extend([p1_tmp, p2_tmp])
        else:
            while len(offs_pop) < self.pop_size*self.pop_split:
                # choosing between crossover and mutation
                p1, p2 = self.selector(population,True), self.selector(population,True)
                    # make sure that the parents are different
                while p1 == p2:
                
                    p1, p2 = self.selector(population,True), self.selector(population,True)
                p1_tmp= copy.deepcopy(p1)
                p2_tmp= copy.deepcopy(p2)    
            
                if random.random() < self.p_xo: # if crossover is selected
                # choose two parents
                
                # generate offspring from the chosen parents
                    offs1, offs2 = self.crossover(
                        p1_tmp.Tree.repr_,
                        p2_tmp.Tree.repr_,
                        tree1_n_nodes=p1_tmp.Tree.node_count,
                        tree2_n_nodes=p2_tmp.Tree.node_count
                    )

                # assuring the offspring do not exceed max_depth
                    if max_depth is not None:
                        while (
                            depth_calculator(offs1) > max_depth
                            or depth_calculator(offs2) > max_depth
                        ):
                            offs1, offs2 = self.crossover(
                                p1_tmp.Tree.repr_,
                                p2_tmp.Tree.repr_,
                                tree1_n_nodes=p1_tmp.Tree.node_count,
                                tree2_n_nodes=p2_tmp.Tree.node_count
                            )
                    p1_tmp=nslug_individual( [1 if key in used_terminals(offs1,individual.TERMINALS) else 0 for key in individual.TERMINALS],offs1)
                    p2_tmp=nslug_individual( [1 if key in used_terminals(offs2,individual.TERMINALS) else 0 for key in individual.TERMINALS],offs2)        
                    
                    # grouping the offspring in a list to be added to the offspring population
                    offspring = [p1_tmp, p2_tmp]

                else: # if mutation was chosen
                    
                    # generating a mutated offspring from the parent
                    offs1 = self.mutator(p1_tmp.Tree.repr_, num_of_nodes=p1_tmp.Tree.node_count)

                    # making sure the offspring does not exceed max_depth
                    if max_depth is not None:
                        while depth_calculator(offs1) > max_depth:
                            offs1 = self.mutator(p1_tmp.Tree.repr_, num_of_nodes=p1_tmp.Tree.node_count)

                    # adding the offspring to a list, to be added to the offspring population
                    p1_tmp=nslug_individual( [1 if key in used_terminals(offs1,individual.TERMINALS) else 0 for key in individual.TERMINALS],offs1)
                
                    offspring = [p1_tmp]

            # adding the offspring as instances of Tree to the offspring population
                offs_pop.extend(offspring)
                
            while len(offs_pop) < self.pop_size:
                # choosing between crossover and mutation
                p1, p2 = self.selector(population,True), self.selector(population,True)
                    # make sure that the parents are different
                while p1 == p2:
                
                    p1, p2 = self.selector(population,True), self.selector(population,True)
                p1_tmp= copy.deepcopy(p1.individual)
                p2_tmp= copy.deepcopy(p2.individual)    
                
                offs1=p1.Tree.repr_
                offs2=p2.Tree.repr_
                
                # choosing between crossover and mutation
                if random.random() < self.p_xo: # if crossover is selected
                # generate offspring from the chosen parents
                    p1_tmp, p2_tmp  = self.ga_crossover(p1_tmp,p2_tmp)
                    
                    offs1=create_random_random_tree(6,self.FUNCTIONS,p1_tmp.chromossome,self.CONSTANTS,0)
                    offs2=create_random_random_tree(6,self.FUNCTIONS,p2_tmp.chromossome,self.CONSTANTS,0)
                if random.random() < self.p_m: # if crossover is selected
                # generate offspring from the chosen parents
                    p1_tmp=self.ga_mutator(p1_tmp)
                    p2_tmp=self.ga_mutator(p2_tmp)
                    
                    offs1=create_random_random_tree(6,self.FUNCTIONS,p1_tmp.chromossome,self.CONSTANTS,0)
                    offs2=create_random_random_tree(6,self.FUNCTIONS,p2_tmp.chromossome,self.CONSTANTS,0)
                    
                p1_tmp=nslug_individual( p1_tmp.encoder,create_random_random_tree(6,self.FUNCTIONS,p1_tmp.chromossome,self.CONSTANTS,0))
                p2_tmp=nslug_individual( p2_tmp.encoder,create_random_random_tree(6,self.FUNCTIONS,p2_tmp.chromossome,self.CONSTANTS,0))                    
            # adding the offspring as instances of Tree to the offspring population
               
                offs_pop.extend([p1_tmp,p2_tmp])
 
        # making sure the offspring population is of the same size as the population
        if len(offs_pop) > population.size:
            offs_pop = offs_pop[: population.size]

        # turning the offspring population into an instance of Population
        offs_pop = nslug_Population(offs_pop)
        offs_pop.evaluate(ffunction, X=X_train, y=y_train, n_jobs=n_jobs)
        # evaluating the offspring population
        for slug_ind in offs_pop.population:
            slug_ind.individual= ga_delete(slug_ind.individual,used_terminals(slug_ind.Tree.repr_,slug_ind.individual.TERMINALS))
       
        # for ind in offs_pop.population:
        #      print(ind.individual.chromossome)
        #      print(dict(sorted(used_terminals(ind.Tree.repr_,individual.TERMINALS).items(), key=lambda item: item[1], reverse=False)) )
        #      print("------------------------------------------------------------")
            

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
                self.elite.Tree.test_fitness,
                self.elite.Tree.node_count,
                float(niche_entropy([ind.Tree.repr_ for ind in population.population])),
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
        elif log == 5:
            for ind in population.population:
                vic_keys = list(ind.individual.TERMINALS.keys())
                logger(
                    log_path,
                    generation,
                    self.elite.Tree.fitness,
                    elapsed_time,
                    float(population.nodes_count),
                    additional_infos = [
                        
                        {vic_keys[i]: ind.individual.TERMINALS[vic_keys[i]] for i in range(len(ind.individual.encoder)) if ind.individual.encoder[i] == 1},  
                        ind.individual.encoder,
                        ind.individual.chromossome,
                        used_terminals(ind.Tree.repr_,ind.individual.TERMINALS),
                        ind.Tree.fitness
                        
                    ],
                    run_info=run_info,
                    seed=self.seed,
                )
            return
        else:
            add_info = [self.elite.Tree.test_fitness, self.elite.Tree.node_count, log]

        logger(
            log_path,
            generation,
            self.elite.Tree.fitness,
            elapsed_time,
            float(population.nodes_count),
            additional_infos=add_info,
            run_info=run_info,
            seed=self.seed,
        )
