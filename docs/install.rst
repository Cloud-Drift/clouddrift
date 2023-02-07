.. _install:

Installation
============

You can install the latest release of CloudDrift using pip or Conda.
You can also install the latest development (unreleased) version from GitHub.

pip
---

In your virtual environment, type:

.. code-block:: text

  pip install clouddrift

Conda
-----

First add ``conda-forge`` to your channels in your Conda environment:

.. code-block:: text

  conda config --add channels conda-forge
  conda config --set channel_priority strict

then install CloudDrift:

.. code-block:: text

  conda install clouddrift

Developers
----------

If you need the latest development version, get it from GitHub using pip:

.. code-block:: text

  pip install git+https://github.com/Cloud-Drift/clouddrift

Running tests
=============

To run the tests, you need to first download the CloudDrift source code from
GitHub and install it in your virtual environment:


.. code-block:: text

  git clone https://github.com/cloud-drift/clouddrift
  cd clouddrift
  python3 -m venv venv
  source venv/bin/activate
  pip install .

Then, run the tests like this:

.. code-block:: text

  python -m unittest tests/*.py

A quick how-to guide is provided on the `Usage <https://cloud-drift.github.io/clouddrift/usage.html>`_ page.