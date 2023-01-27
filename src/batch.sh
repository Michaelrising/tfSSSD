#HK SE
#nohup python3 -u train-csdi.py --algo 'transformer' --cuda 1 --stock 'SE' > ../log/stocks/train_csdi_se_transformer.log  2>&1 &    OK
#nohup python3 -u train-csdi.py --algo 'S4' --cuda 0 --stock 'SE' > ../log/stocks/train_csdi_se_s4.log  2>&1 &  OK
#nohup python3 -u train-csdi.py --algo 'S5' --cuda 1 --stock 'SE'> ../log/stocks/train_csdi_se_s5.log  2>&1 & OK
#nohup python3 -u train-csdi.py --algo 'Mega' --cuda 1 --batch_size 32 --stock 'SE'  > ../log/stocks/train_csdi_se_mega.log  2>&1 &

# HK DJ
#nohup python3 -u train-csdi.py --algo 'transformer' --cuda 1 --stock 'DJ' > ../log/stocks/train_csdi_dj_transformer.log  2>&1 & OK
#nohup python3 -u train-csdi.py --algo 'S4' --cuda 0 --stock 'DJ' > ../log/stocks/train_csdi_dj_s4.log  2>&1 & OK
#nohup python3 -u train-csdi.py --algo 'S5' --cuda 1 --stock 'DJ'> ../log/stocks/train_csdi_dj_s5.log  2>&1 & OK
#nohup python3 -u train-csdi.py --algo 'Mega' --cuda 0 --batch_size 32 --stock 'DJ'  > ../log/stocks/train_csdi_dj_mega.log  2>&1 & OK

# HK ES
#nohup python3 -u train-csdi.py --algo 'transformer' --cuda 0 --stock 'ES' > ../log/stocks/train_csdi_es_transformer.log  2>&1 & OK
#nohup python3 -u train-csdi.py --algo 'S4' --cuda 0 --stock 'ES' > ../log/stocks/train_csdi_es_s4.log  2>&1 & OK
#nohup python3 -u train-csdi.py --algo 'S5' --cuda 0 --stock 'ES'> ../log/stocks/train_csdi_es_s5.log  2>&1 & OK
#nohup python3 -u train-csdi.py --algo 'Mega' --cuda 1 --batch_size 32 --stock 'ES'  > ../log/stocks/train_csdi_es_mega.log  2>&1 &




# SZ
#nohup python3 -u train-sssd.py --cuda 0 --alg "S4" --stock 'DJ' > ../log/stocks/train_sssd_dj_s4.log  2>&1 & OK
#nohup python3 -u train-sssd.py --cuda 0 --alg "S5" --stock 'DJ'  > ../log/stocks/train_sssd_dj_s5.log  2>&1 &
#nohup python3 -u train-sssd.py --cuda 1 --alg "Mega" --stock 'DJ' > ../log/stocks/train_sssd_dj_mega.log  2>&1 & OK

# SZ
#nohup python3 -u train-sssd.py --cuda 0 --alg "S4" --stock 'ES' > ../log/stocks/train_sssd_es_s4.log  2>&1 &     OK
#nohup python3 -u train-sssd.py --cuda 1 --alg "S5" --stock 'ES'  > ../log/stocks/train_sssd_es_s5.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 1 --alg "Mega" --stock 'ES' > ../log/stocks/train_sssd_es_mega.log  2>&1 &


# HK
# nohup python3 -u train-sssd.py --cuda 0 --alg "S4" --stock 'SE' > ../log/stocks/train_sssd_se_s4.log  2>&1 &      OK

# SZ
#nohup python3 -u train-sssd.py --cuda 1 --alg "S5" --stock 'SE'  > ../log/stocks/train_sssd_se_s5.log  2>&1 &
#nohup python3 -u train-sssd.py --cuda 0 --alg "Mega" --stock 'SE' > ../log/stocks/train_sssd_se_mega.log  2>&1 &



nohup python3 -u evaluate_sssd.py --algo 'S4' --cuda 1 --stock 'DJ' > ../log/stocks/evaluate_csdi_s4.log  2>&1 &
nohup python3 -u evaluate_sssd.py --algo 'Mega' --cuda 0 --data 'stocks' > ../log/stocks/evaluate_csdi_mega.log  2>&1 &
nohup python3 -u evaluate_sssd.py --algo 'S5' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_s5.log  2>&1 &





nohup python3 -u evaluate_csdi.py --algo 'transformer' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_transformer.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'S4' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_s4.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'Mega' --cuda 0 --data 'stocks' > ../log/stocks/evaluate_csdi_mega.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'S5' --cuda 1 --data 'stocks' > ../log/stocks/evaluate_csdi_s5.log  2>&1 &
