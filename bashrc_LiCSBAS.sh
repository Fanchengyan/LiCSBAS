## Path
LICSBAS_PATH="$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)"
export LICSBAS_PATH=$LICSBAS_PATH
export PATH="$LICSBAS_PATH/bin:$PATH"
export PYTHONPATH="$LICSBAS_PATH/LiCSBAS_lib:$PYTHONPATH"

## Add to .bashrc
echo '# LICSBAS' >> ~/.bashrc
echo 'export LICSBAS_PATH='$LICSBAS_PATH >> ~/.bashrc
echo 'export PATH="$LICSBAS_PATH/bin:$PATH"' >> ~/.bashrc
echo 'export PYTHONPATH="$LICSBAS_PATH/LiCSBAS_lib:$PYTHONPATH"' >> ~/.bashrc


