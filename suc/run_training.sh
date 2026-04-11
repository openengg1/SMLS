#!/bin/bash
cd /home/rmishra/projects/stochasticMLSpray/model/suc
export PYTHONDONTWRITEBYTECODE=1
python3 train_supervised_cluster_routing.py 2>&1 | tee training_shell.log
