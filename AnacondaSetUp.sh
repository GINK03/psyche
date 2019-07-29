wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh 
sh Anaconda3-2019.07-Linux-x86_64.sh -b -p 
#conda install -c conda-forge opencv -y
pip install opencv-python
pip install optuna
conda install pytorch torchvision -c pytorch -y
conda install -c conda-forge lightgbm -y
conda install -c conda-forge xgboost -y
conda install -c conda-forge altair -y
conda install -c conda-forge catboost -y
conda install -c conda-forge pandas-profiling -y
