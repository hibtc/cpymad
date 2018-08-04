"""
Parse command line options.
"""

from os import environ


def remove_arg(args, opt, envname):
    """
    Remove one occurence of ``--PARAM=VALUE`` or ``--PARAM VALUE`` from
    ``args`` and return the corresponding values.
    """
    for i, arg in enumerate(args):
        if arg == opt:
            del args[i]
            return args.pop(i)
        elif arg.startswith(opt + '='):
            del args[i]
            return arg.split('=', 1)[1]
    return environ.get(envname)


def remove_opt(args, opt, envname, default=None):
    """Remove one occurence of ``--PARAM`` or ``--no-PARAM`` from ``args``
    and return the corresponding boolean value."""
    if opt in args:
        args.remove(opt)
        return True
    neg = '--no' + opt[1:]
    if neg in args:
        args.remove(neg)
        return False
    return bool(int(environ[envname])) if envname in environ else default
