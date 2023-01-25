nohup python3 -u train-csdi.py --algo 'S4' --cuda 0 > ../log/stocks/train_csdi_s4.log  2>&1 &
nohup python3 -u train-csdi.py --algo 'S5' --cuda 1 > ../log/stocks/train_csdi_s5.log  2>&1 &
nohup python3 -u train-csdi.py --algo 'Mega' --cuda 0 > ../log/stocks/train_csdi_mega.log  2>&1 &
nohup python3 -u train-csdi.py --algo 'transformer' --cuda 1 > ../log/stocks/train_csdi_transformer.log  2>&1 &




nohup python3 -u evaluate_csdi.py --algo 'transformer' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_transformer.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'S4' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_s4.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'Mega' --cuda 0 --data 'stocks' > ../log/stocks/evaluate_csdi_mega.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'S5' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_s5.log  2>&1 &


nohup python3 -u train-csdi.py > ../log/mujoco/CSDI-S4/train_csdis4.log  2>&1 &

nohup python3 -u train-sssd.py > ../log/stocks/train_sssd_s4.log  2>&1 &
nohup python3 -u train-sssd.py > ../log/stocks/train_sssd_s5.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 1 > ../log/stocks/train_sssd_mega.log  2>&1 &