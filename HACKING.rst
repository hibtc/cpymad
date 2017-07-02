Hacking
~~~~~~~

Try to be consistent with the PEP8_ guidelines. Add `unit tests`_ for all
non-trivial functionality. `Dependency injection`_ is a great pattern to
keep modules testable.

Commits should be reversible, independent units if possible. Use descriptive
titles and also add an explaining commit message unless the modification is
trivial. See also: `A Note About Git Commit Messages`_.

.. _PEP8: http://www.python.org/dev/peps/pep-0008/
.. _`unit tests`: http://docs.python.org/2/library/unittest.html
.. _`Dependency injection`: http://www.youtube.com/watch?v=RlfLCWKxHJ0
.. _`A Note About Git Commit Messages`: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html

Tests
~~~~~

The `Travis CI`_ service is mainly used to check that the unit tests for
cpymad itself execute on several python versions. Python{2.7,3.3} are
supported. The tests are executed on any update of an upstream branch.
The Travis builds use a unofficial precompiled libmadx-dev_ package to
avoid having to rebuild the entire MAD-X library on each invocation.

.. _`Travis CI`: https://travis-ci.org/hibtc/cpymad
.. _libmadx-dev: https://github.com/hibtc/madx-debian
