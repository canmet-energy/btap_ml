Installation
============

This project implements a pipeline designed to run on both \*nix and Windows based systems.
Some elements of this documentation may be specific to running on the
`DAaaS <https://statcan.github.io/daaas/>`_ environment. If you will be working on AAW, you may want to refer to
:doc:`aaw_setup` to configure the environment first.

To download and utilize the Docker image of the project, consult the documentation on the
`Docker image's page <https://hub.docker.com/r/juliantemp/btap_ml>`_. The content below is only when the project
is being worked on without Docker. To utilize your GPU with the docker image, you will need to ensure that your
computer is configured to pass your GPU through to Docker with the appropriate drivers.

Python
------

A minimum of Python 3.8+ is required to reliably run this project.

.. _dependency-install:

Dependencies
------------

Basic dependencies required to run the simulations are listed in `requirements.txt`. Open a terminal window and
run ``pip`` to install the requirements::

    pip install -r requirements.txt

Additional dependencies exist if you would like to participate in the development of the system::

    pip install -r requirements-dev.txt

.. warning::

   If the code is being run in an environment with existing dependencies, a new environment should be defined to avoid conflicts.

Pre-commit hooks
^^^^^^^^^^^^^^^^

.. note::

   This step is only necessary if you are interested in participating in the development of the model.

.. warning::

   ``pre-commit`` is installed as part of the ``requirements-dev.txt``. Ensure to run that step first.

.. warning::

   You may run into issues with conda environments in which pre-commit fails unless on the base environment.
   This may require ``pre-commit`` to be installed on your base conda environment to work.

The project makes use of ``pre-commit``, which is a package that helps to keep source code clean and consistent
through hooks run against every commit. When these hooks detect something to fix they can cause your commit to fail,
but often the file just needs to be re-added to the commit as a result of changes being applied (fixing whitespace,
for example).

The command that needs to be executed to set up the hooks is::

    pre-commit install

After executing the above command, the pre-commit hooks will run automatically when performing commits.
