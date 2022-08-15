Working on AAW
==============

Getting the project set up on the Advanced Analytics Workspace (AAW) can be achieved by following the guide provided
for `Kubeflow Setup <https://statcan.github.io/daaas/en/1-Experiments/Kubeflow/>`_. Here are some values you can use
when creating the notebook:

* Name: Your name or a descriptive purpose
* Namespace: nrcan-btap
* Image: jupyterlab-cpu
* CPU: 1.0
* Memory: 32.0Gi
* Workspace volume: New (let it create one for you)
  * Size: 16.0Gi
* GPUs: None for basic use OR 1 Nvidia GPU for enhanced use (where the GPU usage may change the parameters above)

It takes a few minutes to create the notebook, but from that point on you can connect to it easily from the
``Notebook Servers`` menu (left side) in the Kubeflow dashboard.

Logging in
----------

When the notebook is ready, you can log in to see the `Jupyter Launcher <https://statcan.github.io/daaas/en/1-Experiments/Jupyter/>`_
screen. From there you can use, open, or create notebooks, or you can launch Visual Studio Code to have an IDE environment.

Project source code
===================

Once you have the machine set up, you can access the project source code by cloning the `git repository <https://github.com/canmet-energy/btap_ml.git>`_. Use the git
icon on the left side of the Jupyter interface to clone the repository.

The repository is available at https://github.com/canmet-energy/btap_ml. Once the clone is complete you can interact
with the files through `the jupyter interface <https://statcan.github.io/daaas/en/1-Experiments/Jupyter/>`_, and use
the launcher interface (top left of interface) to open a new terminal to run the scripts in the project.

Required packages
-----------------

A lot of the packages that are commonly used for data science are included in the environment, but there are always
things that need to be added further. You can install them easily with ``pip`` as outlined at :ref:`dependency-install`.

You are now ready to start working!
