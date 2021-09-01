"""
Implementation of a simple LL(1) parser.
"""
from copy import deepcopy


def fix_point(func, x, **kwargs):
    """
    Repeatedly apply ``x := f(x, **kwargs)`` until encountering a fix
    point, i.e. ``x == f(x)`` and return the resulting fix point ``x``.
    """
    while True:
        x_new = func(x, **kwargs)
        if x_new == x:
            return x
        else:
            x = x_new


def extend_empty_sets(empty: dict, grammar: dict) -> dict:
    """
    Determine which nonterminals of an LL(1) grammar can be empty.
    Must be applied repeatedly until converging.

    :param empty: nonterminal -> bool
    :param grammar: nonterminal -> [productions...]
    :returns: Extended copy of ``empty``
    """
    return {
        symbol: empty.get(symbol) or any(
            all(empty.get(p) for p in production)
            for production in productions
        )
        for symbol, productions in grammar.items()
    }


def extend_firsts_sets(firsts, *, empty, grammar):
    """
    Determine the FIRST sets of an LL(1) grammar, i.e. the set of terminals
    that a given symbol can start with.

    Must be applied repeatedly until converging.

    :param firsts: terminal|nonterminal -> set[terminal]
    :param empty: nonterminal -> bool
    :param grammar: nonterminal -> [productions...]
    :returns: Extended copy of ``firsts``
    """
    firsts = deepcopy(firsts)
    for symbol, productions in grammar.items():
        for production in productions:
            for i, n in enumerate(production):
                extend_parse_table(symbol, firsts[symbol], {
                    t: production[i + 1:][::-1] + p
                    for t, p in firsts[n].items()
                })
                if not empty[n]:
                    break
    return firsts


def extend_follow_sets(follow, *, firsts, empty, grammar):
    """
    Determine the FOLLOW sets of an LL(1) grammar, i.e. the set of terminals
    that a given symbol can be followed by if it may be empty.

    Must be applied repeatedly until converging.

    :param follow: terminal|nonterminal -> set[terminal]
    :param firsts: terminal|nonterminal -> set[terminal]
    :param empty: nonterminal -> bool
    :param grammar: nonterminal -> [productions...]
    :returns: Extended copy of ``follow``
    """
    follow = deepcopy(follow)
    for symbol, productions in grammar.items():
        for production in productions:
            for i, p1 in enumerate(production):
                for p2 in production[i + 1:]:
                    follow[p1] |= set(firsts[p2])
                    if not empty[p2]:
                        break
                else:
                    follow[p1] |= follow[symbol]
    return follow


def create_parse_table(terminals, grammar, start):
    """
    Create an LL(1) parsing table.

    :param terminals: list of terminal symbols
    :param grammar: nonterminal -> [productions...]
    :param start: nonterminal start symbol
    :returns: parse table nonterminal -> {terminal -> production}
    """
    empty = fix_point(extend_empty_sets, {}, grammar=grammar)
    empty.update({t: False for t in terminals})
    table = {**{t: {t: []} for t in terminals}, **{n: {} for n in grammar}}
    table = fix_point(
        extend_firsts_sets,
        table,
        empty=empty,
        grammar=grammar)
    follow = {
        **{t: set() for t in terminals},
        **{n: set() for n in grammar},
    }
    follow = fix_point(
        extend_follow_sets, follow,
        firsts=table, empty=empty, grammar=grammar)

    for symbol, terminals in follow.items():
        if empty[symbol]:
            extend_parse_table(symbol, table[symbol], {
                t: None
                for t in terminals
            })

    return table


def extend_parse_table(symbol, old, new):
    """
    Merge parse tables ``old`` and ``new`` without silently overriding
    duplicate entries. Raises a ``ValueError`` if ``new`` contains entries
    that were defined differently in ``old``.
    """
    for key in old.keys() & new.keys():
        if old[key] != new[key]:
            raise ValueError((
                "Grammar is ambiguous. Duplicate entry "
                "in parse table: <{}, {}> -> {} or {}"
            ).format(symbol, key, old[key], new[key]))
    old.update(new)


class Parser:

    """
    LL(1) syntax checker.

    :param terminals: list of terminal symbols
    :param grammar: nonterminal -> [productions...]
    :param start: nonterminal start symbol
    """

    def __init__(self, terminals, grammar, start):
        self.table = table = create_parse_table(
            terminals, grammar, start)
        # precompule rules lookup:
        self.rules = {symbol: {} for symbol in table}
        for symbol, rules in table.items():
            self.rules[symbol].update({
                t: p and [self.rules[n] for n in p]
                for t, p in rules.items()
            })
        self.start = self.rules[start]

    def parse(self, tokens):
        """
        Verifies the grammar.

        :param list tokens: list of tokens (terminals) in input order
        :returns: nothing if successful
        :raises: ValueError
        """
        tokens = list(reversed(tokens))
        stack = [self.start]
        try:
            while stack:
                token = tokens[-1]
                rules = stack.pop()
                more = rules[token.type]
                if more is not None:
                    stack.extend(more)
                    tokens.pop()
        except KeyError:
            raise ValueError(
                ("Unexpected {} in:\n"
                 "    {!r}\n"
                 "     ").format(token.type, token.expr)
                + ' ' * token.start
                + '^' * max(token.length, 1)
            ) from None
