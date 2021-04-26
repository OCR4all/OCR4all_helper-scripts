class Record(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
