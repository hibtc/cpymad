# encoding: utf-8
"""
This module defines a convenience layer to access the MAD-X interpreter.

The most interesting class for users is :class:`Madx`.
"""

from __future__ import absolute_import

import logging
import os

import numpy as np

from minrpc.util import ChangeDirectory

from . import _rpc
from . import util
from .object import (
    Sequence, SequenceMap, Command, CommandMap, ArrayAttribute,
    GlobalElementList, BaseTypeMap, TableMap, VarList,
)

try:
    basestring
except NameError:
    basestring = str


__all__ = [
    'Madx',
    'Sequence',
    'CommandLog',
    'metadata',
]


class Version(object):

    """Version information struct. """

    def __init__(self, release, date):
        """Store version information."""
        self.release = release
        self.date = date
        self.info = tuple(map(int, release.split('.')))

    def __repr__(self):
        """Show nice version string to user."""
        return "MAD-X {} ({})".format(self.release, self.date)


class CommandLog(object):

    """Log MAD-X command history to a text file."""

    @classmethod
    def create(cls, filename, prefix='', suffix='\n'):
        """Create CommandLog from filename (overwrite/create)."""
        return cls(open(filename, 'wt'), prefix=prefix, suffix=suffix)

    def __init__(self, file, prefix='', suffix='\n'):
        """Create CommandLog from file instance."""
        self._file = file
        self._prefix = prefix
        self._suffix = suffix

    def __call__(self, command):
        """Log a single history line and flush to file immediately."""
        self._file.write(self._prefix + command + self._suffix)
        self._file.flush()


class Madx(object):

    """
    Python interface for a MAD-X process.

    For usage instructions, please refer to:

        https://hibtc.github.io/cpymad/getting-started

    Communicates with a MAD-X interpreter in a background process.

    The state of the MAD-X interpreter is controlled by feeding textual MAD-X
    commands to the interpreter.

    The state of the MAD-X interpreter is accessed by directly reading the
    values from the C variables in-memory and sending the results pickled back
    over the pipe.

    Data attributes:

    :ivar command:      Mapping of all MAD-X commands.
    :ivar globals:      Mapping of global MAD-X variables.
    :ivar elements:     Mapping of globally visible elements.
    :ivar base_types:   Mapping of MAD-X base elements.
    :ivar sequence:     Mapping of all sequences in memory.
    :ivar table:        Mapping of all tables in memory.
    """

    def __init__(self, libmadx=None, command_log=None, error_log=None,
                 **Popen_args):
        """
        Initialize instance variables.

        :param libmadx: :mod:`libmadx` compatible object
        :param command_log: Log all MAD-X commands issued via cpymad.
        :param error_log: logger instance ``logging.Logger``
        :param Popen_args: Additional parameters to ``subprocess.Popen``

        If ``libmadx`` is NOT specified, a new MAD-X interpreter will
        automatically be spawned. This is what you will mostly want to do. In
        this case any additional keyword arguments are forwarded to
        ``subprocess.Popen``. The most prominent use case for this is to
        redirect or suppress the MAD-X standard I/O::

            m = Madx(stdout=False)

            with open('madx_output.log', 'w') as f:
                m = Madx(stdout=f)

            m = Madx(stdout=subprocess.PIPE)
            f = m._process.stdout
        """
        # get logger
        if error_log is None:
            error_log = logging.getLogger(__name__)
        # open history file
        if isinstance(command_log, basestring):
            command_log = CommandLog.create(command_log)
        # start libmadx subprocess
        if libmadx is None:
            # stdin=None leads to an error on windows when STDIN is broken.
            # Therefore, we need set stdin=os.devnull by passing stdin=False:
            Popen_args.setdefault('stdin', False)
            Popen_args.setdefault('bufsize', 0)
            self._service, self._process = \
                _rpc.LibMadxClient.spawn_subprocess(**Popen_args)
            libmadx = self._service.libmadx
        if not libmadx.is_started():
            libmadx.start()
        # init instance variables:
        self._libmadx = libmadx
        self._command_log = command_log
        self._error_log = error_log
        self.command = CommandMap(self)
        self.globals = VarList(self)
        self.elements = GlobalElementList(self)
        self.base_types = BaseTypeMap(self)
        self.sequence = SequenceMap(self)
        self.table = TableMap(self._libmadx)

    def __bool__(self):
        """Check if MAD-X is up and running."""
        try:
            return self._libmadx.is_started()
        except (_rpc.RemoteProcessClosed, _rpc.RemoteProcessCrashed):
            return False

    __nonzero__ = __bool__      # alias for python2 compatibility

    def __getattr__(self, name):
        """Resolve missing attributes as commands."""
        return getattr(self.command, name)

    # Data descriptors:

    @property
    def version(self):
        """Get the MAD-X version."""
        return Version(self._libmadx.get_version_number(),
                       self._libmadx.get_version_date())

    @property
    def options(self):
        """Values of current options."""
        return Command(self, self._libmadx.get_options())

    # Methods:

    def input(self, text):
        """
        Run any textual MAD-X input.

        :param str text: command text
        """
        # write to history before performing the input, so if MAD-X
        # crashes, it is easier to see, where it happened:
        if self._command_log:
            self._command_log(text)
        try:
            self._libmadx.input(text)
        except _rpc.RemoteProcessCrashed:
            # catch + reraise in order to shorten stack trace (~3-5 levels):
            raise RuntimeError("MAD-X has stopped working!")

    __call__ = input

    def expr_vars(self, expr):
        """Find all variable names used in an expression. This does *not*
        include element attribute nor function names."""
        return [v for v in util.expr_symbols(expr)
                if util.is_identifier(v)
                and v in self.globals
                and self._libmadx.get_var_type(v) > 0]

    def chdir(self, path):
        """
        Change the directory of the MAD-X process (not the current python process).

        :param str path: new path name
        :returns: a context manager that can change the directory back
        :rtype: ChangeDirectory

        It can be used as context manager for temporary directory changes::

            with madx.chdir('/x/y/z'):
                madx.call('file.x')
                madx.call('file.y')

        This method is special in that it is currently the only modification
        of the MAD-X interpreter state that doesn't go through the
        :meth:`Madx.input` method (because there is no MAD-X command to change
        the directory).
        """
        # Note, that the libmadx module includes the functions 'getcwd' and
        # 'chdir' so it can be used as a valid 'os' module for the purposes
        # of ChangeDirectory:
        return ChangeDirectory(path, self._libmadx)

    def call(self, file, chdir=False):
        """
        CALL a file in the MAD-X interpreter.

        :param str file: file name with path
        :param bool chdir: temporarily change directory in MAD-X process
        """
        if chdir:
            dirname, basename = os.path.split(file)
            with self.chdir(dirname):
                self.command.call(file=basename)
        else:
            self.command.call(file=file)

    def twiss(self, **kwargs):
        """
        Run TWISS.

        :param str sequence: name of sequence
        :param kwargs: keyword arguments for the MAD-X command

        Note that the kwargs overwrite any arguments in twiss_init.
        """
        self.command.twiss(**kwargs)
        table = kwargs.get('table', 'twiss')
        if 'file' not in kwargs:
            self._libmadx.apply_table_selections(table)
        return self.table[table]

    def survey(self, **kwargs):
        """
        Run SURVEY.

        :param str sequence: name of sequence
        :param kwargs: keyword arguments for the MAD-X command
        """
        self.command.survey(**kwargs)
        table = kwargs.get('table', 'survey')
        if 'file' not in kwargs:
            self._libmadx.apply_table_selections(table)
        return self.table[table]

    def use(self, sequence=None, range=None, **kwargs):
        """
        Run USE to expand a sequence.

        :param str sequence: sequence name
        :returns: name of active sequence
        """
        self.command.use(sequence=sequence, range=range, **kwargs)

    def sectormap(self, elems, **kwargs):
        """
        Compute the 7D transfer maps (the 7'th column accounting for KICKs)
        for the given elements and return as Nx7x7 array.
        """
        self.command.select(flag='sectormap', clear=True)
        for elem in elems:
            self.command.select(flag='sectormap', range=elem)
        with util.temp_filename() as sectorfile:
            self.twiss(sectormap=True, sectorfile=sectorfile, **kwargs)
        return self.sectortable(kwargs.get('sectortable', 'sectortable'))

    def sectortable(self, name='sectortable'):
        """Read sectormap + kicks from memory and return as Nx7x7 array."""
        tab = self.table[name]
        cnt = len(tab['r11'])
        return np.vstack((
            np.hstack((tab.rmat(slice(None)),
                       tab.kvec(slice(None)).reshape((6,1,-1)))),
            np.hstack((np.zeros((1, 6, cnt)),
                       np.ones((1, 1, cnt)))),
        )).transpose((2,0,1))

    def sectortable2(self, name='sectortable'):
        """Read 2nd order sectormap T_ijk, return as Nx6x6x6 array."""
        tab = self.table[name]
        return tab.tmat(slice(None)).transpose((3,0,1,2))

    def match(self,
              constraints=[],
              vary=[],
              weight=None,
              method=('lmdif', {}),
              knobfile=None,
              limits=None,
              **kwargs):
        """
        Perform a simple MATCH operation.

        For more advanced cases, you should issue the commands manually.

        :param list constraints: constraints to pose during matching
        :param list vary: knob names to be varied
        :param dict weight: weights for matching parameters
        :param str knobfile: file to write the knob values to
        :param dict kwargs: keyword arguments for the MAD-X command
        :returns: final knob values
        :rtype: dict

        Example:

        >>> from cpymad.madx import Madx
        >>> from cpymad.types import Constraint
        >>> m = Madx()
        >>> m.call('sequence.madx')
        >>> twiss_init = {'betx': 1, 'bety': 2, 'alfx': 3, 'alfy': 4}
        >>> m.match(
        ...     sequence='mysequence',
        ...     constraints=[
        ...         dict(range='marker1',
        ...              betx=Constraint(min=1, max=3),
        ...              bety=2)
        ...     ],
        ...     vary=['qp1->k1',
        ...           'qp2->k1'],
        ...     **twiss_init,
        ... )
        >>> tw = m.twiss('mysequence', **twiss_init)
        """
        command = self.command
        # MATCH (=start)
        command.match(**kwargs)
        if weight:
            command.weight(**weight)
        for c in constraints:
            command.constraint(**c)
        limits = limits or {}
        for v in vary:
            command.vary(name=v, **limits.get(v, {}))
        command[method[0]](**method[1])
        command.endmatch(knobfile=knobfile)
        return dict((knob, self.eval(knob)) for knob in vary)

    def verbose(self, switch=True):
        """Turn verbose output on/off."""
        self.command.option(echo=switch, warn=switch, info=switch)

    def eval(self, expr):
        """
        Evaluates an expression and returns the result as double.

        :param str expr: expression to evaluate.
        :returns: numeric value of the expression
        :rtype: float
        """
        if isinstance(expr, (float, int, bool)):
            return expr
        if isinstance(expr, (list, ArrayAttribute)):
            return [self.eval(x) for x in expr]
        # Try to prevent process crashes:
        # NOTE: this limits to a sane subset of accepted MAD-X expressions.
        util.check_expression(expr)
        return self._libmadx.eval(expr)


class Metadata(object):

    """MAD-X metadata (license info, etc)."""

    __title__ = u'MAD-X'

    @property
    def __version__(self):
        return self._get_libmadx().get_version_number()

    __summary__ = (
        u'MAD is a project with a long history, aiming to be at the '
        u'forefront of computational physics in the field of particle '
        u'accelerator design and simulation. The MAD scripting language '
        u'is de facto the standard to describe particle accelerators, '
        u'simulate beam dynamics and optimize beam optics.'
    )

    __support__ = u'mad@cern.ch'

    __uri__ = u'http://madx.web.cern.ch/madx/'

    __credits__ = (
        u'MAD-X is developed at CERN and has many contributors. '
        u'For more information see:\n'
        u'\n'
        u'http://madx.web.cern.ch/madx/www/contributors.html'
    )

    def get_copyright_notice(self):
        from pkg_resources import resource_string
        return resource_string('cpymad', 'COPYING/madx.rst').decode('utf-8')

    _libmadx = None

    def _get_libmadx(self):
        if not self._libmadx:
            svc, proc = _rpc.LibMadxClient.spawn_subprocess()
            self._libmadx = svc.libmadx
        return self._libmadx


metadata = Metadata()
