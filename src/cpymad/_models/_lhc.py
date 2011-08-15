from _model import model

class lhc(model):
    ##
    # Initialize object
    def __init__(self,optics="collision"):
        model.__init__(self,'lhc',optics)