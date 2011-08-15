


def runj():
    import jpymad
    run(jpymad)
def runc():
    import cpymad
    run(cpymad)

def run(pymad):
    l=pymad.model('lhc')
    print("Available sequences: "+str(l.list_sequences()))
    # would it be possible to implement
    # same type of return here?
    t,p=l.twiss('lhcb1')

if __name__=="__main__":
    #runj()
    runc()