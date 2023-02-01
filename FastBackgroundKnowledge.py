from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.Node import Node
class FastBackgroundKnowledge(BackgroundKnowledge):
    def __init__(self):
        super().__init__()
        self.dict_forbidden = {}
        self.dict_required = {}

    def forbidden_rules_to_dict(self):
        forbDict = {}
        for forb in self.forbidden_rules_specs:
            if forb[0].get_name() in forbDict.keys():
                new_val = forbDict[forb[0].get_name()]
                new_val.append(forb[1].get_name())
                forbDict[forb[0].get_name()] = new_val
            else:
                forbDict[forb[0].get_name()] = [forb[1].get_name()]

        self.dict_forbidden= forbDict

    def required_rules_to_dict(self):
        reqDict = {}
        for req in self.required_rules_specs:
            if req[0].get_name() in reqDict.keys():
                new_val = reqDict[req[0].get_name()]
                new_val.append(req[1].get_name())
                reqDict[req[0].get_name()] = new_val
            else:
                reqDict[req[0].get_name()] = [req[1].get_name()]

        self.dict_required = reqDict

    def is_forbidden(self, node1: Node, node2: Node) -> bool:
        """
        check whether the edge node1 --> node2 is forbidden

        Parameters
        ----------
        node1: the from node in edge which is checked
        node2: the to node in edge which is checked

        Returns
        -------
        if the  edge node1 --> node2 is forbidden, then return True, otherwise False.
        """
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError('node1 and node2 must be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(
                type(node2)))

        if node1.get_name() in self.dict_forbidden.keys():
            values = self.dict_forbidden[node1.get_name()]
            if (node2.get_name() in values):
                return True

        return False

    def is_required(self, node1: Node, node2: Node) -> bool:
        """
        check whether the edge node1 --> node2 is required

        Parameters
        ----------
        node1: the from node in edge which is checked
        node2: the to node in edge which is checked

        Returns
        -------
        if the  edge node1 --> node2 is required, then return True, otherwise False.
        """

        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError('node1 and node2 must be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(
                type(node2)))

        if node1.get_name() in self.dict_required.keys():
            values = self.dict_required[node1.get_name()]
            if (node2.get_name() in values):
                return True
        return False