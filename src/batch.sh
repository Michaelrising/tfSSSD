nohup python3 -u train-csdi.py --algo 'S4' --cuda 0 > ../log/stocks/train_csdi_s4.log  2>&1 &
nohup python3 -u train-csdi.py --algo 'S5' --cuda 1 > ../log/stocks/train_csdi_s5.log  2>&1 &
nohup python3 -u train-csdi.py --algo 'Mega' --cuda 0 --batch_size 32  > ../log/stocks/train_csdi_mega.log  2>&1 &
nohup python3 -u train-csdi.py --algo 'transformer' --cuda 1 > ../log/stocks/train_csdi_transformer.log  2>&1 &


nohup python3 -u evaluate_csdi.py --algo 'transformer' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_transformer.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'S4' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_s4.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'Mega' --cuda 0 --data 'stocks' > ../log/stocks/evaluate_csdi_mega.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'S5' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_s5.log  2>&1 &




nohup python3 -u train-sssd.py --cuda 1 --alg "S4" --stock 'DJ' > ../log/stocks/train_sssd_dj_s4.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 0 --alg "S5" --stock 'DJ'  > ../log/stocks/train_sssd_dj_s5.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 1 --alg "Mega" --stock 'DJ' > ../log/stocks/train_sssd_dj_mega.log  2>&1 &