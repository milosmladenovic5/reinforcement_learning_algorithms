import numpy as np
from sumtree import SumTree

class Memory(object): #stored as (s, a, r, s_) in SumTree

    PER_e = 0.01 # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6 # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4 # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1. #clipped abs error

    def __init__(self, capacity):
        #making the tree
        #the tree is composed of a sum tree that contains priority and data array


        self.tree = SumTree(capacity)

    def store(self, experience):
        #find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience) # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, n):
        # Create a sample array that will contain the minibatch
        memory_batch = []

        b_idx, b_ISWeights = np.empty((n, ), dtype = np.int32), np.empty((n, 1), dtype = np.float32)

        # Calculate the priority segment
        # here, as explained in the paper, we divide the Range[0, prior_total] into n ranges
        priority_segment = self.tree.total_priority / n #priority segment

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling]) # max = 1

        #Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            b_idx[i]= index
            
            experience = [data]
            
            memory_batch.append(experience)
        
        return b_idx, memory_batch, b_ISWeights
