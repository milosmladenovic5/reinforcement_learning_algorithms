import numpy as np

class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity #number of leaf nodes (final nodes) that contain
        #experiences

        self.tree = np.zeros(2*capacity - 1) # 1 for the root node
        self.data = np.zeros(capacity, dtype = object)

    def add(self, priority, data):
        #look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity -1

        #update data frame
        self.data[self.data_pointer] = data

        #update the leaf
        self.update(tree_index, priority)

        self.data_pointer+=1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0 

    #update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # change = new_priority_score - former_priority_score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index-1)//2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]

