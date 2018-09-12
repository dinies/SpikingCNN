# SpikingCNN
Build a spiking convolutional neural network in tensorflow for image recognitioon tasks




Guide to set up the environment,
tested on linux 18.04LTS

pyenv and virtualenv set up :


sudo apt install curl git-core gcc make zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libssl-dev
git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv

Add at .bashrc file

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
eval "$(pyenv init -)"
fi

Source it:
source $HOME/.bashrc

Virtualenv plug in:
git clone https://github.com/yyuu/pyenv-virtualenv.git   $HOME/.pyenv/plugins/pyenv-virtualenv
source $HOME/.bashrc

Remainder of useful pyenv commands:
pyenv install -l
pyenv versions
pyenv virtualenv 3.6.5 venv_name
pyenv activate venv_name
pyenv deactivate
pyenv commands

Our set up: install 3.6.5 and create a virtualenv for tensorflow

