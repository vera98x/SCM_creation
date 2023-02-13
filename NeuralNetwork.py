from causallearn.graph.GeneralGraph import GeneralGraph

class NN_sample:
    def __init__(self, label, prev_event = 0, prev_platform = 0, dep1 = 0, dep2 = 0):
        self.label = label
        self.prev_event = prev_event
        self.prev_platform = prev_platform
        self.dep1 = dep1
        self.dep2 = dep2

