

from matplotlib import pyplot as plt
import pymad as pm



def plot_beta(model, postfix=''):
    # Run twiss on the model, optionally give name of file 
    # where tfs table is stored
    result, summary = model.twiss(seqname='lhcb1', columns=['name', 's', 'betx', 'bety'], file='lhcb1' + postfix + '.tfs')
    
    plt.figure()
    # do something with the result (note, madx is still waiting!):
    plt.xlabel('dist. from IP1')
    plt.ylabel(r'$\beta$')
    plt.plot(result.s, result.betx, label=r'$\beta_x$')
    plt.plot(result.s, result.bety, label=r'$\beta_y$')
    plt.savefig('beta' + postfix + '.eps')


# choose the mode
mode = 'jpymad'
#mode = 'jpymad'

# some hacks for the moment:
# we have not the same model definitions in both implementations at the moment
if mode is 'jpymad':
    mdefname = 'LHC (LSA)'
    opticname = 'A55C55A1000L500_0.00900_2011'
else:
    mdefname = '???'
    opticname = 'collision'

#
# Here it starts
#
# create the service
pms = pm.init(mode)

# print the name of all model definitions
print pms.mdefnames

# alternatively use convenience function
#pm.ls_mdefs()

# get one model-definition
mdef = pms.get_mdef(mdefname)
model = pms.create_model(mdef)

# alternatively: create the model by name and get the model definition afterwards:
# model = pms.create_model(mdefname)
# mdef = model.mdef

# list the available (running) models
pm.ls_models()

# print a list of available sequences:
print mdef.seqnames 

plot_beta(model, '_inj')

# list the available optics and set a new one
print mdef.opticnames
model.set_optic(opticname)

plot_beta(model, '_coll') 

# remove the model from the service:
pms.delete_model(model)

# and do a cleanup
pm.cleanup()
