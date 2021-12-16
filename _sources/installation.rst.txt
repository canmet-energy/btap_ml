Installation
============

This project implements a pipeline designed to run on \*nix based systems, and as such may have requirements not
documented here to be able to run on Windows. Some elements of this documentation may be specific to running on the
`DAaaS <https://statcan.github.io/daaas/>`_ environment. If you will be working on AAW, you may want to refer to
:doc:`aaw_setup` to configure the environment first.

Python
------

A minimum of Python 3.8+ is required to reliably run this project.

.. _dependency-install:

Dependencies
------------

Basic dependencies required to run the simulations are listed in `requirements.txt`. From the terminal, run the command below and these can be installed with::

    pip install -r requirements.txt

Additional dependencies exist if you would like to participate in the development of the system::

    pip install -r requirements-dev.txt

Pre-commit hooks
^^^^^^^^^^^^^^^^

.. note::

   This step is only necessary if you are interested in participating in the development of the model.

The project makes use of ``pre-commit``, which is a package that helps to keep source code clean and consistent
through hooks run against every commit. When these hooks detect something to fix they can cause your commit to fail,
but often the file just needs to be readded to the commit as a result of changes being applied (fixing whitespace,
for example).

The only thing that needs to happen to set up the hooks is to run::

    pre-commit install

From that point on things will run automatically when performing commits as normal.
