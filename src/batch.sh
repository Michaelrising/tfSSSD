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
#nohup python3 -u train-csdi.py --algo 'Mega' --cuda 1 --batch_size 32 --stock 'ES'  > ../log/stocks/train_csdi_es_mega.log  2>&1 & OK





#nohup python3 -u train-sssd.py --cuda 0 --alg "S4" --stock 'DJ' > ../log/stocks/DJ/train_sssd_dj_s4.log  2>&1 & DONE
#nohup python3 -u train-sssd.py --cuda 1 --alg "S5" --stock 'DJ'  > ../log/stocks/DJ/train_sssd_dj_s5.log  2>&1 & DONE
#nohup python3 -u train-sssd.py --cuda 0 --alg "Mega" --stock 'DJ' > ../log/stocks/DJ/train_sssd_dj_mega.log  2>&1 & DONE


#nohup python3 -u train-sssd.py --cuda 1 --alg "S4" --stock 'ES' > ../log/stocks/ES/train_sssd_es_s4.log  2>&1 & DONE
#nohup python3 -u train-sssd.py --cuda 1 --alg "S5" --stock 'ES'  > ../log/stocks/ES/train_sssd_es_s5.log  2>&1 & DONE
#nohup python3 -u train-sssd.py --cuda 0 --alg "Mega" --stock 'ES' > ../log/stocks/ES/train_sssd_es_mega.log  2>&1 & DONE



#nohup python3 -u train-sssd.py --cuda 0 --alg "S4" --stock 'SE' > ../log/stocks/SE/train_sssd_se_s4.log  2>&1 & DONE
#nohup python3 -u train-sssd.py --cuda 0 --alg "Mega" --stock 'SE' > ../log/stocks/SE/train_sssd_se_mega.log  2>&1 & DONE
#nohup python3 -u train-sssd.py --cuda 1 --alg "S5" --stock 'SE'  > ../log/stocks/SE/train_sssd_se_s5.log  2>&1 & DONE


nohup python3 -u train-sssd.py --cuda 0 --alg "S4" --stock 'all' --batch_size 2 > ../log/stocks/all/train_sssd_all_s4.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 0 --alg "S5" --stock 'all' --batch_size 15 > ../log/stocks/all/train_sssd_all_s5.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 1 --alg "Mega" --stock 'all' --batch_size 3 > ../log/stocks/all/train_sssd_all_mega.log  2>&1 &


########################################################################################################################


#nohup python3 -u evaluate_sssd.py --algo 'S4' --cuda 0 --stock 'DJ' > ../log/stocks/DJ/evaluate_sssd_dj_s4.log  2>&1 & # DONE
#nohup python3 -u evaluate_sssd.py --algo 'S5' --cuda 0 --stock 'DJ' > ../log/stocks/DJ/evaluate_sssd_dj_s5.log  2>&1 & # DONE
nohup python3 -u evaluate_sssd.py --algo 'Mega' --cuda 1 --stock 'DJ' > ../log/stocks/DJ/evaluate_sssd_dj_mega.log  2>&1 & #


#nohup python3 -u evaluate_sssd.py --algo 'S4' --cuda 1 --stock 'ES' > ../log/stocks/ES/evaluate_sssd_es_s4.log  2>&1 & # DONE
#nohup python3 -u evaluate_sssd.py --algo 'S5' --cuda 0 --stock 'ES' > ../log/stocks/ES/evaluate_sssd_es_s5.log  2>&1 & # DONE
nohup python3 -u evaluate_sssd.py --algo 'Mega' --cuda 1 --stock 'ES' > ../log/stocks/ES/evaluate_sssd_es_mega.log  2>&1 & #



#nohup python3 -u evaluate_sssd.py --algo 'S4' --cuda 0 --stock 'SE' > ../log/stocks/SE/evaluate_sssd_se_s4.log  2>&1 & # DONE
#nohup python3 -u evaluate_sssd.py --algo 'S5' --cuda 0 --stock 'SE' > ../log/stocks/SE/evaluate_sssd_se_s5.log  2>&1 & # DONE
nohup python3 -u evaluate_sssd.py --algo 'Mega' --cuda 0 --stock 'SE' > ../log/stocks/SE/evaluate_sssd_se_mega.log  2>&1 & #






nohup python3 -u evaluate_sssd.py --algo 'S5' --cuda 0 --stock 'SE' > ../log/stocks/evaluate_sssd_se_s5.log  2>&1 &












#nohup python3 -u evaluate_csdi.py --algo 'transformer' --cuda 1  --stock 'DJ' > ../log/stocks/evaluate_csdi_dj_transformer.log  2>&1 & OK
#nohup python3 -u evaluate_csdi.py --algo 'S4' --cuda 0 --stock 'DJ' > ../log/stocks/evaluate_csdi_dj_s4.log  2>&1 & OK
nohup python3 -u evaluate_csdi.py --algo 'Mega' --cuda 0 --stock 'DJ' > ../log/stocks/evaluate_csdi_dj_mega.log  2>&1 &
#nohup python3 -u evaluate_csdi.py --algo 'S5' --cuda 1 --stock 'DJ' > ../log/stocks/evaluate_csdi_dj_s5.log  2>&1 &


nohup python3 -u evaluate_csdi.py --algo 'transformer' --cuda 1  --stock 'ES' > ../log/stocks/evaluate_csdi_dj_transformer.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'S4' --cuda 0 --stock 'ES' > ../log/stocks/evaluate_csdi_dj_s4.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'Mega' --cuda 0 --stock 'ES' > ../log/stocks/evaluate_csdi_dj_mega.log  2>&1 &
nohup python3 -u evaluate_csdi.py --algo 'S5' --cuda 1 --stock 'ES' > ../log/stocks/evaluate_csdi_dj_s5.log  2>&1 &



