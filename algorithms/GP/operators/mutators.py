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
Mutator operator implementation.
"""

import random
import math
import numpy as np
from nslug.utils.utils import create_custom_list
from nslug.algorithms.GP.representations.tree import Tree
from nslug.algorithms.GP.representations.tree_utils import (create_grow_random_tree,
                                                                random_subtree,
                                                                substitute_subtree, tree_depth,create_full_random_tree_list_mode,used_terminals,find_all_paths,replace_at_path,last_node_depth,last_parent_depth)


# Function to perform mutation on a tree.
def mutate_tree_node(max_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c):
    """
    Generates a function for mutating a node within a tree representation based on a set of
    terminals, constants, and functions.

    This function returns another function that can mutate a specific node in the tree representation.
    The mutation process involves randomly choosing between modifying a terminal, constant, or function node,
    while ensuring the resulting tree representation maintains valid arity (i.e., the number of child nodes
    expected by the function node).

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree to consider during mutation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    p_c : float
        Probability of choosing a constant node for mutation.

    Returns
    -------
    Callable
        A function ('m_tn') that performs subtree mutation within a tree representation.

        The mutation process involves randomly choosing between modifying a terminal, constant, or function node,
        while ensuring the resulting tree representation maintains valid arity (i.e., the number of child nodes
        expected by the function node). Depending on the maximum depth of the tree or the size of the original, the
        mutation process may only return a single node.

        Parameters
        ----------
        tree : tuple
            The tree representation to mutate.

        Returns
        -------
        tuple
            The structure of the mutated tree representation.
        str
            The node resulting from mutation

    Notes
    -----
    The returned function (`m_tn`) operates recursively to traverse the tree representation and
    randomly select a node for mutation.
    """
    def m_tn(tree):
        """
        Performs subtree mutation within a tree representation.

        The mutation process involves randomly choosing between modifying a terminal, constant, or function node,
        while ensuring the resulting tree representation maintains valid arity (i.e., the number of child nodes
        expected by the function node). Depending on the maximum depth of the tree or the size of the original, the
        mutation process may only return a single node.

        Parameters
        ----------
        tree : tuple
            The tree representation to mutate.

        Returns
        -------
        tuple
            The structure of the mutated tree representation.
        str
            The node resulting from mutation
        """
        # if the maximum depth is one or the tree is just a terminal, choose a random node
        if max_depth <= 1 or not isinstance(tree, tuple):
            # choosing between a constant and a terminal
            if random.random() > p_c:
                return np.random.choice(list(TERMINALS.keys()))
            else:
                return np.random.choice(list(CONSTANTS.keys()))

        # randomly choosing a node to mutate based on the arity
        if FUNCTIONS[tree[0]]["arity"] == 2:
            node_to_mutate = np.random.randint(0, 3)
        elif FUNCTIONS[tree[0]]["arity"] == 1:
            node_to_mutate = np.random.randint(0, 2)  #

        # obtaining the mutating function
        inside_m = mutate_tree_node(max_depth - 1, TERMINALS, CONSTANTS, FUNCTIONS, p_c)

        # if the first node is to be mutated
        if node_to_mutate == 0:
            new_function = np.random.choice(list(FUNCTIONS.keys()))
            it = 0

            # making sure the arity of the chosen function matches the arity of the function to be mutated
            while (
                FUNCTIONS[tree[0]]["arity"] != FUNCTIONS[new_function]["arity"]
                or tree[0] == new_function
            ):
                new_function = np.random.choice(list(FUNCTIONS.keys()))

                it += 1
                # if a new valid function was not found in 10 tries, return the original function
                if it >= 10:
                    new_function = tree[0]
                    break

            # mutating the left side of the tree
            left_subtree = inside_m(tree[1])

            # mutating the right side of the tree, if the arity is 2
            if FUNCTIONS[tree[0]]["arity"] == 2:
                right_subtree = inside_m(tree[2])
                return new_function, left_subtree, right_subtree
            # if the arity is 1, returning the new function and the modified left tree
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                return new_function, left_subtree

        # if the node to mutate is in position 1
        elif node_to_mutate == 1:
            # preserving the node in position 0 and 2 while mutating position 1
            left_subtree = inside_m(tree[1])
            if FUNCTIONS[tree[0]]["arity"] == 2:
                return tree[0], left_subtree, tree[2]
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                return tree[0], left_subtree
        # if the node to mutate is in position 2
        else:
            # preserving the node in position 0 and 1 while mutating position 2
            right_subtree = inside_m(tree[2])
            return tree[0], tree[1], right_subtree

    return m_tn


def mutate_tree_subtree(max_depth, TERMINALS, CONSTANTS, FUNCTIONS, p_c):
    """
    Generates a function for performing subtree mutation within a tree representation.

    This function returns another function that can perform subtree mutation by selecting a random subtree
    in the tree representation and replacing it with a newly generated random subtree.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree to consider during mutation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    p_c : float
        Probability of choosing a constant node for mutation.

    Returns
    -------
    Callable
        A function ('innee_mur') that mutates a subtree in the given tree representation by replacing a randomly
        selected subtree.

        This function selects a random subtree in the input tree representation and substitutes it
        with a newly generated random subtree of the same maximum depth. If a terminal is passed,
        returns the original.

        Parameters
        ----------
        tree1 : tuple or str
            The tree representation to mutate.
        num_of_nodes : int, optional
            The number of nodes in the tree, used for selecting a random subtree.

        Returns
        -------
        tuple
            The mutated tree representation with a new subtree
        str
            The original terminal node if the input was a terminal

    Notes
    -----
    The returned function (`inner_mut`) operates by selecting a random subtree from the input tree
    representation and replacing it with a randomly generated tree representation of the same maximum depth.
    """
    # getting the subtree substitution function and the random subtree selection function
    subtree_substitution = substitute_subtree(FUNCTIONS=FUNCTIONS)
    random_subtree_picker = random_subtree(FUNCTIONS=FUNCTIONS)
    last_node_depth_picker = last_node_depth(FUNCTIONS=FUNCTIONS)
    last_parent_depth_picker = last_parent_depth(FUNCTIONS=FUNCTIONS)    

    def inner_mut(tree1, num_of_nodes=None, c=0):
        """
        Mutates a subtree in the given tree representation by replacing a randomly selected subtree.

        This function selects a random subtree in the input tree representation and substitutes it
        with a newly generated random subtree of the same maximum depth. If a terminal is passed,
        returns the original.

        Parameters
        ----------
        tree1 : tuple or str
            The tree representation to mutate.
        num_of_nodes : int, optional
            The number of nodes in the tree, used for selecting a random subtree.

        Returns
        -------
        tuple
            The mutated tree representation with a new subtree
        str
            The original terminal node if the input was a terminal
        """
        if isinstance(tree1, tuple): # if the tree is a base (gp) tree
            mutation_point = random_subtree_picker(
                tree1, num_of_nodes=num_of_nodes
            )
            
            #print(tree_depth(Tree.FUNCTIONS)(mutation_point))
            #subtree_depth=random.randint(1,tree_depth(Tree.FUNCTIONS)(mutation_point))

            
            if isinstance(tree1, tuple):
                a=last_node_depth_picker(tree1,mutation_point)
                n_max=17-last_node_depth_picker(tree1,mutation_point)+1
            else:
                a=last_parent_depth_picker(tree1,mutation_point)
                n_max=17-tree_depth(Tree.FUNCTIONS)(tree1)
            
            if c>0:
                print(tree_depth(Tree.FUNCTIONS)(tree1))
                print(mutation_point)
                print(tree_depth(Tree.FUNCTIONS)(mutation_point))
                print(f'no inicio da ultima arvore: {a}')
                print(f'profundidade permitida: {n_max}')
            subtree_depth=random.randint(1,max(n_max,1))
            #max_depth
            
            
            # gettubg a bew subtree
            new_subtree = create_grow_random_tree(
                subtree_depth, FUNCTIONS, TERMINALS, CONSTANTS, p_c=p_c
            )
            # replacing the tree in mutation point for the new substring
            new_tree1 = subtree_substitution(
                tree1, mutation_point, new_subtree
            )
            if c>0:
                print("-----------------------arvore original----------------------")
                print(tree1)
                print("--------------depth arvore original----------------")
                print(tree_depth(Tree.FUNCTIONS)(tree1))
                print("--------------ponto corte----------------")
                print(mutation_point)
                print("--------------depth ponto corte----------------")
                print(tree_depth(Tree.FUNCTIONS)(mutation_point))
                print("--------------nova subarvore----------------")
                print(new_subtree)
                print("--------------depth nova subarvore----------------")
                print(tree_depth(Tree.FUNCTIONS)(new_subtree))
                print("--------------arvore final----------------")
                print(new_tree1)
                print("--------------depth arvore final----------------")
                print(tree_depth(Tree.FUNCTIONS)(new_tree1))
            return new_tree1
        else:
            return tree1 # if tree1 is a terminal
    return inner_mut


def gp_insert(tree,used_terminals):
    
    tree_depth=math.ceil(math.log(len(used_terminals), 2))
    terminal_list=create_custom_list(list(used_terminals.keys()),tree_depth)
    terminal_list2=terminal_list.copy()
    new_tree=Tree(create_full_random_tree_list_mode(tree_depth+1,tree.FUNCTIONS,terminal_list2))

    return new_tree

def gp_insert_add(algorithm,TERMINALS, FUNCTIONS):
    
    if algorithm=="update":
        def gp_insert(tree,new_terminals):
            
            used_terminal=used_terminals(tree.repr_, TERMINALS)
            diff_keys = set(new_terminals) - set(used_terminal)
            expr=tree.repr_
            for z in diff_keys:
                y = random.choice(list(used_terminal.keys()))
                paths = find_all_paths(tree.repr_, y)
                if not paths:
                    continue
                chosen_path = random.choice(paths)
                func_name = random.choice(list(FUNCTIONS.keys()))
                new_subtree = (func_name, y, z)
                expr = replace_at_path(expr, chosen_path, new_subtree)
            new_tree=Tree(expr)

            return new_tree
        return gp_insert
    if algorithm=="create_new":
        def gp_insert(tree,used_terminals):
            
            tree_depth=math.ceil(math.log(len(used_terminals), 2))
            terminal_list=create_custom_list(list(used_terminals.keys()),tree_depth)
            terminal_list2=terminal_list.copy()
            new_tree=Tree(create_full_random_tree_list_mode(tree_depth+1,tree.FUNCTIONS,terminal_list2))

            return new_tree
        return gp_insert
    