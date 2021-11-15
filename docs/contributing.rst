.. _contributing:

Contributing
============

Where to start?
---------------

Bug reports and requests
------------------------

Contributing to the documentation
---------------------------------

Install sphinx and the Read the Docs theme:

.. code-block:: text

  conda install sphinx
  pip install sphinx-rtd-theme

Then from the ``clouddrift/docs`` directory, run:

.. code-block:: text

  make html

to create the static website stored in ``clouddrift/docs/_build/html/``. The main page ``index.html`` can be visualized in any browser. After modifying the documentation, it might be necessary to run ``make clean`` before rebuilding.

Using pytest
------------
