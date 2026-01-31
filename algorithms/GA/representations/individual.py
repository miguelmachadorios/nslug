
"""
Tree class implementation for representing tree structures in genetic programming.
"""

from nslug.algorithms.GP.representations.tree_utils import bound_value, flatten, tree_depth, _execute_tree
import torch

class individual:
    """
    The Tree class representing the candidate solutions in genetic programming.

    Attributes
    ----------
    repr_ : tuple or str
        Representation of the tree structure.
    FUNCTIONS : dict
        Dictionary of allowed functions in the tree representation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree representation.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree representation.
    depth : int
        Depth of the tree.
    fitness : float
        Fitness value of the tree.
    test_fitness : float
        Test fitness value of the tree.
    node_count : int
        Number of nodes in the tree.
    """

    TERMINALS = None
  
    def __init__(self, encoder):
        """
        Initializes a Tree object.

        Parameters
        ----------
        repr_ : tuple
            Representation of the tree structure.
        """
        #self.TERMINALS = TERMINALS
        self.TERMINALS =  individual.TERMINALS
        vic_keys = list(self.TERMINALS.keys())
        self.encoder = encoder
        self.fitness = None
        self.test_fitness = None
        self.used_variables = encoder.count(1)
        self.chromossome={vic_keys[i]: self.TERMINALS[vic_keys[i]] for i in range(len(encoder)) if encoder[i] == 1}
        self.model_repr_= None

    
    def evaluate(self, ffunction, X, y, testing=False, new_data = False, caller_id= None ):
        """
        Evaluates the tree given a fitness function, input data (X), and target data (y).

        Parameters
        ----------
        ffunction : function
            Fitness function to evaluate the individual.
        X : torch.Tensor
            The input data (which can be training or testing).
        y : torch.Tensor
            The expected output (target) values.
        testing : bool, optional
            Flag indicating if the data is testing data. Default is False.
        new_data : bool, optional
            Flag indicating that the input data is new and the model is being used outside the training process.

        Returns
        -------
        None
            If the data is training or testing data, the fitness value is attributed to the individual.
        float
            If exposed to new data, the fitness value is returned.
        """
        # getting the predictions (i.e., semantics) of the individual
        preds = self.apply_tree(X)

        # if new (testing data) is being used, return the fitness value for this new data
        if new_data:
            return float(ffunction(y, preds))

        # if not, attribute the fitness value to the individual
        else:
            if testing:
                self.test_fitness = ffunction(y, preds)
            else:
                self.fitness = ffunction(y, preds)

    