
"""
Mutator operator implementation.
"""

import random

from nslug.algorithms.GA.representations.individual import individual
import numpy as np


def ga_mutation(indiv):

    while True:
        mutated_individual = [1 - value if random.random() <= 1/len(indiv.encoder) else value 
                   for value in indiv.encoder]
        if sum(mutated_individual) > 0:
            return individual(mutated_individual)

def ga_delete(indiv,used_variables):
    
    indices_to_zero = [indiv.chromossome[key] for key in indiv.chromossome.keys() - used_variables.keys()]
    for i in indices_to_zero:
        indiv.encoder[i]=0
    return individual(indiv.encoder)


def ga_insert(indiv):
    p=1/len(indiv.TERMINALS)
    return individual([1 if (value == 0 and random.random() < p) else value 
            for value in indiv.encoder])