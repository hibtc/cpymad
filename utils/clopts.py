"""
Parse command line options.
"""

from os import environ


def remove_arg(args, name):
    """
    Remove one occurence of ``--PARAM=VALUE`` or ``--PARAM VALUE`` from
    ``args`` and return the corresponding values.
    """
    opt = '--' + name
    for i, arg in enumerate(args):
        if arg == opt:
            del args[i]
            return args.pop(i)
        elif arg.startswith(opt + '='):
            del args[i]
            return arg.split('=', 1)[1]
    return environ.get(name.upper())


def remove_opt(args, name, default=None):
    """Remove one occurence of ``--PARAM`` or ``--no-PARAM`` from ``args``
    and return the corresponding boolean value."""
    opt = '--' + name
    if opt in args:
        args.remove(opt)
        return True
    neg = '--no-' + name
    if neg in args:
        args.remove(neg)
        return False
    envname = name.upper()
    return bool(int(environ[envname])) if envname in environ else default
