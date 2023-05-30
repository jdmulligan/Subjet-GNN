#!/usr/bin/bash

# On hiccup CPU, we use pyenv -- on hiccup GPU, we will use system python
if ! lspci | grep -i 'nvidia' > /dev/null; then
    # Set up pyenv (for python version management)
    export PYENV_ROOT="/home/software/users/james/pyenv"
    export PYTHON_CONFIGURE_OPTS="--enable-shared"
    export PATH="${PATH}:${PYENV_ROOT}/bin"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    pyenv local 3.9.12
fi

# Get command line option to determine whether we need to install the virtual environment, or just enter it
for i in "$@"; do
  case $i in
    --install)
      INSTALL=TRUE
      shift
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done
if [ ! -z ${INSTALL} ]; then
    echo
    echo "Create new virtual environment..."
    if lspci | grep -i 'nvidia' > /dev/null; then
        python -m venv venv_gpu
        echo "Nvidia GPU detected...installing requirements_gpu.txt"
        pip install -r requirements_gpu.txt
    else
        python -m venv venv_cpu
        echo "No Nvidia GPU detected...installing requirements_cpu.txt"
        pip install -r requirements_cpu.txt
    fi
fi

# Initialize python virtual environment for package management
if lspci | grep -i 'nvidia' > /dev/null; then
    source venv_gpu/bin/activate
else
    source venv_cpu/bin/activate
fi
