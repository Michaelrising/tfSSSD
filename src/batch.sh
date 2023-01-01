nohup python3 -u train-csdi.py > ../log/mujoco/CSDI/train_csdi.log  2>&1 &
nohup python3 -u evaluate_csdi.py --cuda 1  > ../log/mujoco/CSDI/evaluate_csdi.log  2>&1 &

nohup python3 -u train-csdi.py > ../log/mujoco/CSDI-S4/train_csdis4.log  2>&1 &