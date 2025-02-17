Installation
============

Langesim is available on PyPI and can be installed with pip. To install the
package, create a virtual environment (see below), activate it, then install with pip:

    .. code-block:: console

        pip install langesim

Alternatively, the package can be installed from source:        

1. Clone the *langesim* repository

    .. code-block:: console

        git clone https://github.com/GabrielTellez/langesim.git

2. To install the package you have two options: install with poetry or with pip.
   First, enter the top directory of the package 

    .. code-block:: console 

        cd langesim

    - If you have poetry installed (https://python-poetry.org/), run 

    .. code-block:: console

        poetry install

    - Otherwise, pip can be used to install the package. Create a virtual
      environment, activate it, then install with pip:

    .. code-block:: console

        python -m venv .langesim_env 
        
        # Activate environment: 
        # bash shell
        source .langesim_env/bin/activate
        # or windows cmd
        .langesim_env\Scripts\activate.bat
        # or windows powershell
        .langesim_env\Scripts\activate.ps1

        pip install .


