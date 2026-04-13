class _TransitionBuilder:
    def __init__(self, source, label):
        self.source = source
        self.label = label

    def __rshift__(self, target):
        self.source.transitions[self.label] = target
        return target


class Node:
    def __init__(self):
        self.default_next = None
        self.transitions = {}

    def prep(self, shared):
        return shared

    def exec(self, prep_res):
        return prep_res

    def post(self, shared, prep_res, exec_res):
        return None

    def __rshift__(self, target):
        self.default_next = target
        return target

    def __sub__(self, label):
        return _TransitionBuilder(self, label)


class Flow:
    def __init__(self, start):
        self.start = start

    def run(self, shared):
        current = self.start
        while current is not None:
            prep_res = current.prep(shared)
            exec_res = current.exec(prep_res)
            transition = current.post(shared, prep_res, exec_res)
            if transition is None or transition == "default":
                current = current.default_next
            else:
                current = current.transitions.get(transition)
