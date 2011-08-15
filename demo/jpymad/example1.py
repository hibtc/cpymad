

from matplotlib import pyplot as plt
from jpymad.service import JPyMadService

# start an instance, optionally give name for
# the madx script with commands we have executed.
# m=madx('history.madx')
## no seperate instantiation for a madx-instance

# create the service
useJPyMad = True
if useJPyMad:
    pms = JPyMadService()
else:
    # create a CPyMadService
    pass



# a model needs madx instance as input..
# optionally give name of optics to use
## l=lhc(m,'injection')
l = pms.create_model('LHC (LSA)')

# print list of available sequences:
#print l.getSequences()
print l.get_mdef() # TODO: implement get_seq

# Run twiss on one of them, optionally give name of file 
# where tfs table is stored
#tab,param=m.twiss('LHCB1',fname='lhcb1_inj.tfs')
# TODO implement set active range

# do something with the result (note, madx is still waiting!):
plt.xlabel('dist. from IP1')
plt.ylabel('beta')
plt.plot(tab['S'],tab['BETX'],label='bx')
plt.plot(tab['S'],tab['BETY'],label='by')
plt.savefig('beta.eps')

# do something else...
print l.getOptics()
l.setOptics(m,'collision')
m.twiss('lhcb2',columns='name,s,betx,bety',fname='lhcb2_coll.tfs')

# close instance (not really needed):
m.finish()
