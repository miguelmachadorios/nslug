
"""
Crossover operator implementation.
"""

import numpy as np
import copy
import random
from nslug.algorithms.GA.representations.individual import individual

def single_point_crossover_old(indiv1,indiv2):

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
            crossover_points = np.random.randint(1,len(indiv1.encoder)) 
            
            encoder1=indiv1_tmp.encoder[:-crossover_points] + indiv2_tmp.encoder[-crossover_points:]
            encoder2=indiv2_tmp.encoder[:-crossover_points] + indiv1_tmp.encoder[-crossover_points:]

            if sum(encoder1) > 0 and sum(encoder2) > 0 :
                return individual(encoder1),individual(encoder2)

        
        
def single_point_crossover_smart(indiv1,indiv2):

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
        
        if sum(indiv1.encoder)==1 and sum(indiv2.encoder)==1: return indiv1,indiv2
        
        first_index_p1 = indiv1.encoder.index(1)
        first_index_p2 = indiv2.encoder.index(1)
        last_idx_p1=len(indiv1.encoder) - 1 - indiv1.encoder[::-1].index(1)
        last_idx_p2=len(indiv2.encoder) - 1 - indiv2.encoder[::-1].index(1)
        
        first_idx = next(i for i, (x, y) in enumerate(zip(indiv1.encoder, indiv2.encoder)) if x == 1 or y == 1)
        
        last_idx = len(indiv1.encoder) - 1 - next(i for i, (x, y) in enumerate(zip(reversed(indiv1.encoder), reversed(indiv2.encoder))) if x == 1 or y == 1)

        if first_index_p2>last_idx_p1:
            
            crossover_points = random.choice( list(range(0,last_idx_p1)) + list(range(first_index_p2 + 1, len(indiv1.encoder))))
            
        elif first_index_p1>last_idx_p2:
            
            crossover_points = random.choice( list(range(0,last_idx_p2)) + list(range(first_index_p1 + 1, len(indiv1.encoder))))
        else:
            
            crossover_points = np.random.randint(first_idx,last_idx) 
            
        encoder1=indiv1_tmp.encoder[:crossover_points] + indiv2_tmp.encoder[crossover_points:]
        encoder2=indiv2_tmp.encoder[:crossover_points] + indiv1_tmp.encoder[crossover_points:]

        return individual(encoder1),individual(encoder2),crossover_points



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
        
       
        
        A_prefix = np.cumsum(indiv1_tmp.encoder[::-1])[::-1] > 0       
        B_prefix = np.cumsum(indiv2_tmp.encoder[::-1])[::-1] > 0       
        A_suffix = np.cumsum(indiv1_tmp.encoder) > 0                   
        B_suffix = np.cumsum(indiv2_tmp.encoder) > 0                   

      
        A_pref = A_suffix[:-1]
        A_suf  = A_prefix[1:]
        B_pref = B_suffix[:-1]
        B_suf  = B_prefix[1:]

       
        child1_valid = A_pref | B_suf
        child2_valid = B_pref | A_suf

        valid_points = np.where(child1_valid & child2_valid)[0] + 1  
        
        if valid_points.size == 0:
        
            return indiv1_tmp,indiv2_tmp
        
        crossover_points = random.choice(valid_points)
        
        encoder1=indiv1_tmp.encoder[:crossover_points] + indiv2_tmp.encoder[crossover_points:]
        encoder2=indiv2_tmp.encoder[:crossover_points] + indiv1_tmp.encoder[crossover_points:]

        return individual(encoder1),individual(encoder2)

