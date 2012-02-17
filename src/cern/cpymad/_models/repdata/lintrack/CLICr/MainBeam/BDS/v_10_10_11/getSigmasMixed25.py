#!/afs/cern.ch/user/r/rtomas/lintrack/Python-2.5_32bit/Python-2.5_32bit/bin/python

# Just to make sure that the path to the libraires is defined 
import sys
sys.path.append('/afs/cern.ch/eng/sl/lintrack/Python_Classes4MAD/')


import math
#import copy
import sys
from os import system
#from simplex import Simplex
from mapclass25  import *
import random
from metaclass25 import *

# BDS start condition with active match shouldnt b used since alfas are non zero
#betx=18.38257132
#bety=64.4507753

#these ones are the original ones
betx=66.14532014
bety=17.92472388
#============================

#FFIN conditions with L*=6m lsca =1.4
#betx=64.9998842
#bety=17.99971386
#============================
gamma=2935427.005
ex=660*1e-9
ey=20*1e-9


sigmaFFS=[sqrt(ex*betx/gamma), sqrt(ex/betx/gamma), sqrt(ey*bety/gamma), sqrt(ey/bety/gamma), 0.015]

file='fort.18'
map=Map(1,file)
#tw=twiss("twiss.scaled")
#IPi=tw.indx["IP"]
print ";"
print "sigmax1=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay1=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx1=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy1=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"

map=Map(2,file)
#tw=twiss("twiss")
#IPi=tw.indx["IP"]
print ";"
print "sigmax2=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay2=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx2=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy2=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"

map=Map(3,file)
#tw=twiss("twiss")
#IPi=tw.indx["IP"]
print ";"
print "sigmax3=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay3=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx3=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy3=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"

map=Map(4,file)
print ";"
print "sigmax4=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay4=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx4=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy4=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"

map=Map(5,file)
print ";"
print "sigmax5=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay5=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx5=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy5=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"

map=Map(6,file)
print ";"
print "sigmax6=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay6=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx6=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy6=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"

map=Map(7,file)
print ";"
print "sigmax7=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay7=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx7=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy7=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"

map=Map(8,file)
print ";"
print "sigmax8=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay8=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx8=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy8=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"

map=Map(9,file)
print ";"
print "sigmax9=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay9=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx9=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy9=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"

map=Map(10,file)
print ";"
print "sigmax10=",sqrt(map.sigma('x',sigmaFFS)-map.offset('x',sigmaFFS)**2),";"
print "sigmay10=",sqrt(map.sigma('y',sigmaFFS)-map.offset('y',sigmaFFS)**2),";"
print "sigmapx10=",sqrt(map.sigma('px',sigmaFFS)-map.offset('px',sigmaFFS)**2),";"
print "sigmapy10=",sqrt(map.sigma('py',sigmaFFS)-map.offset('py',sigmaFFS)**2),";"
sys.exit()

#print "beta_x=",tw.BETX[IPi],";"
#print "beta_y=",tw.BETY[IPi],";"
#print "alfa_y=",tw.ALFY[IPi],";"
#print "D_y=",tw.DY[IPi],";"
print "corryx=",map.correlation('y','x',sigmaFFS)-(map.offset('y',sigmaFFS)*map.offset('x',sigmaFFS)),";"
print "corrypx=",map.correlation('py','x',sigmaFFS)-(map.offset('py',sigmaFFS)*map.offset('x',sigmaFFS)),";"
print "corrpyx=",map.correlation('y','px',sigmaFFS)-(map.offset('y',sigmaFFS)*map.offset('px',sigmaFFS)),";"
print "corrpypx=",map.correlation('py','px',sigmaFFS)-(map.offset('py',sigmaFFS)*map.offset('px',sigmaFFS)),";"
print "corryd=",map.correlation('y','d',sigmaFFS),";"
