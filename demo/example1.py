

from madx import madx
from pymadx.models import lhc
from matplotlib import pyplot as plt

# start an instance, optionally give name for
# the madx script with commands we have executed.
m=madx('history.madx')

# a model needs madx instance as input..
# optionally give name of optics to use
l=lhc(m,'injection')

# print list of available sequences:
print l.getSequences()

# Run twiss on one of them, optionally give name of file 
# where tfs table is stored
tab,param=m.twiss('LHCB1',fname='lhcb1_inj.tfs')

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
