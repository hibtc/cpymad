
import os


def modelList():
    pymadloc=os.path.dirname(__file__)
    modelloc=os.path.join(pymadloc,'_models')
    ret=[]
    for f in os.listdir(modelloc):
        if len(f)>5 and f[-5:].lower()=='.json':
            ret.append(f[:-5])
    return ret