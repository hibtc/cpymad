
cern.madx
=========

Several users have been most interested in having direct access to Mad-X inside Python, instead of using models.

The following example is similar to the LHC default example found at /afs/cern.ch/eng/lhc/optics/V6.503/job.sample.madx::

    from cern import madx

    # create a Mad-X running instance:
    m=madx.madx(histfile='job.sample.madx')
    # Note, the commands that were sent to Mad-X are written to the file job.sample.madx

    # Turn off echoing:
    m.command('option,-echo')

    # load the sequence/strength files:
    afs='/afs/cern.ch/eng/lhc/optics/V6.503/'
    m.call(afs+'V6.5.seq')
    m.call(afs+'V6.5.inj.str')

    # set variables in IR on/off:
    vars_on=['X1', 'X2', 'X5', 'X8',
        'SEP1', 'SEP2', 'SEP5', 'SEP8']
    vars_off=['ATLAS', 'ALICE', 'CMS', 'LHCB']
    for var in vars_on:
        m.command('ON_'+var+':=1')
    for var in vars_off:
        m.command('ON_'+var+':=0')

    # set beam parameters:
    m.command('beam, sequence=LHCB1, particle=PROTON, pc=450')

    # call twiss:
    table,parameters=m.twiss('LHCB1',columns=['s','name','betx','bety'])

    # plot table:
    import pylab
    pylab.plot(table.s,table.betx,label=r'$beta_x$')
    pylab.plot(table.s,table.bety,label=r'$beta_y$')
    pylab.legend()
    pylab.show()

