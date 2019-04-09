# SpikingCNN
Build a spiking convolutional neural network in tensorflow for image recognitioon tasks


Instructions on how to run the network. 

Different types of execution are possible:

1) Run directly the classifier with the best dataset and weights obtained

2) Run the entire network with the best weights obtained

3) Run the entire network from scratch

#########################################################################

- Go to "src" folder
- Write on your terminal:

$  python main.py 1  (if you want to run the execution type 1)

$  python main.py 2  (if you want to run the execution type 2)

$  python main.py 3  (if you want to run the execution type 3)







Be careful to:
 - The .sh scripts in the 'scripts' folder could need the execution permission
   If there are related problems use $ chmod -x script.sh on every script
 - Additional information about the executions can be found in the 'log' folder inside the csv file 
 - The main.py is made in such a way that at every execution the logs and other execution related files are cleaned.



























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

