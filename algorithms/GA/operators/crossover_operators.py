
"""
Crossover operator implementation.
"""

import numpy
import copy
import random
from nslug.algorithms.GA.representations.individual import individual

def single_point_crossover(indiv1,indiv2):

        """
        Applies single-point crossover between pairs of parents.
        This function selects a random point at which crossover occurs between the parents, generating offspring.

        Parameters:
            parents (array-like): The parents to mate for producing the offspring.
            offspring_size (int): The number of offspring to produce.

        Returns:
            array-like: An array containing the produced offspring.
        """
        indiv1_tmp= copy.deepcopy(indiv1)
        indiv2_tmp= copy.deepcopy(indiv2)
        

        while True:
            crossover_points = numpy.random.randint(1,len(indiv1.encoder)) 
            encoder1=indiv1_tmp.encoder[:-crossover_points] + indiv2_tmp.encoder[-crossover_points:]
            encoder2=indiv2_tmp.encoder[:-crossover_points] + indiv1_tmp.encoder[-crossover_points:]
          
            if sum(encoder1) > 0 and sum(encoder2) > 0 :
                return individual(encoder1),individual(encoder2)

        
        
