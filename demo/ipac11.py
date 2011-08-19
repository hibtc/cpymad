# This is the example we published in our IPAC'11 paper.

from matplotlib import pyplot as plt
import pymad as pm

# choose the mode
mode = 'cpymad' # 'cpymad'

# currently lacking same model/optic names
if mode is 'jpymad':
    mdefname = 'LHC (LSA)'
elif mode is 'cpymad':
    mdefname = 'lhc'

# create the service
pms = pm.init(mode)

# print the name of all model definitions
print pms.mdefnames()

# get one model-definition
model = pms.create_model(mdefname)

# list the available (running) models
pm.ls_models()

# obtain get twiss table in a python object
table,parameters=model.twiss(columns=['name', 's', 'betx', 'bety'])

# plot the result
plt.plot(table.s,table.betx)
plt.show()