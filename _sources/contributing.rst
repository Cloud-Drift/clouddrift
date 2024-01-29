.. _contributing:

Contributing
============

This project follows `NumFOCUS <https://numfocus.org/code-of-conduct>`_ code of conduct, the short version is:

- Be kind to others. Do not insult or put down others. Behave professionally. Remember that harassment and sexist, racist, or exclusionary jokes are not appropriate.
- All communication should be appropriate for a professional audience including people of many different backgrounds. Sexual language and imagery is not appropriate.
- We are dedicated to providing a harassment-free community for everyone, regardless of gender, sexual orientation, gender identity and expression, disability, physical appearance, body size, race, or religion.
- We do not tolerate harassment of community members in any form.

Thank you for helping make this a welcoming, friendly community for all.

Bug reports and requests
------------------------

We encourage users to participate in the development of CloudDrift by filling out bug reports, participating in discussions, and requesting features on `https://github.com/Cloud-Drift/clouddrift <https://github.com/Cloud-Drift/clouddrift>`_.

Contributing to the documentation
---------------------------------

We also welcome contributions to improving the documentation of the project. To create a local static documentation website, the following packages are required:

.. code-block:: console

  conda install sphinx
  pip install sphinx_book_theme
  pip install sphinx-copybutton

Then, from the ``clouddrift/docs`` directory, run

.. code-block:: console

  make html

to compile and output the documentation website to ``clouddrift/docs/_build/html/``. The index page ``index.html`` can be visualized in a web browser. Note that after modifying the documentation, it might be necessary to run ``make clean`` before rebuilding.
