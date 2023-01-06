nohup python3 -u train-csdi.py --algo 'S4' --cuda 1 > ../log/mujoco/train_csdi_s4.log  2>&1 &
nohup python3 -u train-csdi.py --algo 'transformer' --cuda 0 > ../log/mujoco/train_csdi_transformer.log  2>&1 &


nohup python3 -u evaluate_csdi.py --algo 'transformer' --cuda 0  --model_loc '20230105-151317' > ../log/mujoco/evaluate_csdi_transformer.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'S4' --cuda 1  --model_loc '20230105-151317' > ../log/mujoco/evaluate_csdi_s4.log  2>&1 &


nohup python3 -u train-csdi.py > ../log/mujoco/CSDI-S4/train_csdis4.log  2>&1 &