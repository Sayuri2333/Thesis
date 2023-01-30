#!bin/bash

# for all directories in this HDFS dicrectory: /opt/module/hh
for model in DQN DRQN ConvTransformer Conv_Transformer MFCA OnlyMultiscale
do
python PPO.py --model $model
done
