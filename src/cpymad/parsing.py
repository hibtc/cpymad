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
            for p in production:
                firsts[symbol] |= firsts[p]
                if not empty[p]:
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
                    follow[p1] |= firsts[p2]
                    if not empty[p2]:
                        break
                else:
                    follow[p1] |= follow[symbol]
    return follow


def optimize_table(table: dict) -> dict:
    """
    Inline definitions in a LL(1) parsing table.
    """
    table = deepcopy(table)
    for symbol, rules in table.items():
        for terminal, production in rules.items():
            if production:
                if production[-1] != terminal:
                    production[-1:] = table[production[-1]][terminal]
    return table


def create_parse_table(terminals, grammar, start):
    """
    Create an LL(1) parsing table.

    :param terminals: list of terminal symbols
    :param grammar: nonterminal -> [productions...]
    :param start: nonterminal start symbol
    :returns: parse table nonterminal -> {terminal -> production}
    """
    empty = fix_point(extend_empty_sets, {}, grammar=grammar)
    empty |= {t: False for t in terminals}
    firsts = {t: {t} for t in terminals} | {n: set() for n in grammar}
    firsts = fix_point(
        extend_firsts_sets,
        firsts,
        empty=empty,
        grammar=grammar)
    follow = (
        {t: set() for t in terminals} |
        {n: set() for n in grammar}
    )
    follow = fix_point(
        extend_follow_sets, follow,
        firsts=firsts, empty=empty, grammar=grammar)

    table = {n: {} for n in grammar}

    for symbol, productions in grammar.items():
        for production in productions:
            trigger = set()
            for p in production:
                trigger |= firsts[p]
                if not empty[p]:
                    break
            else:
                trigger |= follow[symbol]

            for terminal in trigger:
                if terminal in table[symbol]:
                    raise ValueError((
                        "Grammar may be ambiguous. Duplicate entry "
                        "in parse table: <{}, {}> -> {}"
                    ).format(symbol, terminal, production))
                else:
                    table[symbol][terminal] = production[::-1]

    return fix_point(optimize_table, table) | {
        terminal: {terminal: [terminal]}
        for terminal in terminals
    }


class Parser:

    """
    LL(1) syntax checker.

    :param terminals: list of terminal symbols
    :param grammar: nonterminal -> [productions...]
    :param start: nonterminal start symbol
    """

    def __init__(self, terminals, grammar, start):
        self.table = create_parse_table(
            terminals, grammar, start)
        self.start = start

    def parse(self, tokens):
        """
        Verifies the grammar.

        :param list tokens: list of tokens (terminals) in input order
        :returns: nothing if successful
        :raises: ValueError
        """
        tokens = list(reversed(tokens))
        table = self.table
        stack = [self.start]
        while stack:
            symbol = stack.pop()
            token = tokens[-1]
            try:
                more = table[symbol][token.type]
            except KeyError:
                raise ValueError(
                    f"Unexpected token {token} for {symbol}!") from None
            if more:
                stack.extend(more[:-1])
                tokens.pop()
