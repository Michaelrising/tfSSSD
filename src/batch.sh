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


nohup python3 -u train-sssd.py --cuda 0 --alg "S4" --stock 'all' --batch_size 16 --seq_len 100 > ../log/stocks/all/train_sssd_all_s4_seq_100_step_lr.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 1 --alg "S4" --stock 'all' --batch_size 16 --seq_len 200 > ../log/stocks/all/train_sssd_all_s4_seq_200_step_lr.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 1 --alg "S4" --stock 'all' --batch_size 8 --seq_len 400 > ../log/stocks/all/train_sssd_all_s4_seq_400_step_lr.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 1 --alg "S4" --stock 'all' --batch_size 2 --seq_len 800 > ../log/stocks/all/train_sssd_all_s4_seq_800_step_lr.log  2>&1 &


nohup python3 -u train-csdi.py --cuda 1 --alg "S4" --stock 'all' --batch_size 16 --seq_len 100 > ../log/stocks/all/train_csdi_all_s4_seq_100.log  2>&1 &
nohup python3 -u train-csdi.py --cuda 1 --alg "S4" --stock 'all' --batch_size 16 --seq_len 200 > ../log/stocks/all/train_csdi_all_s4_seq_200.log  2>&1 &
nohup python3 -u train-csdi.py --cuda 1 --alg "S4" --stock 'all' --batch_size 8 --seq_len 400 > ../log/stocks/all/train_csdi_all_s4_seq_400.log  2>&1 &
nohup python3 -u train-csdi.py --cuda 0 --alg "S4" --stock 'all' --batch_size 4 --seq_len 800 > ../log/stocks/all/train_csdi_all_s4_seq_800.log  2>&1 &


nohup python3 -u evaluate_csdi.py --cuda 0 --alg "S4" --stock 'all' --batch_size 32 --seq_len 400 > ../log/stocks/all/evaluate_csdi_all_s4_seq_400.log  2>&1 &
nohup python3 -u evaluate_csdi.py --cuda 0 --alg "S4" --stock 'all' --batch_size 64 --seq_len 100 > ../log/stocks/all/evaluate_csdi_all_s4_seq_100.log  2>&1 &
nohup python3 -u evaluate_csdi.py --cuda 0 --alg "S4" --stock 'all' --batch_size 64 --seq_len 200 > ../log/stocks/all/evaluate_csdi_all_s4_seq_200.log  2>&1 &
nohup python3 -u evaluate_csdi.py --cuda 0 --alg "S4" --stock 'all' --batch_size 16 --seq_len 800 > ../log/stocks/all/evaluate_csdi_all_s4_seq_800.log  2>&1 &


nohup python3 -u evaluate_sssd.py --cuda 0 --alg "S4" --stock 'all' --batch_size 64 --seq_len 100 > ../log/stocks/all/evaluate_sssd_all_s4_seq_100.log  2>&1 &
nohup python3 -u evaluate_sssd.py --cuda 1 --alg "S4" --stock 'all' --batch_size 64 --seq_len 200 > ../log/stocks/all/evaluate_sssd_all_s4_seq_200.log  2>&1 &
nohup python3 -u evaluate_sssd.py --cuda 0 --alg "S4" --stock 'all' --batch_size 16 --seq_len 400 > ../log/stocks/all/evaluate_sssd_all_s4_seq_400.log  2>&1 &
nohup python3 -u evaluate_sssd.py --cuda 0 --alg "S4" --stock 'all' --batch_size 16 --seq_len 800 --num_layers 20 > ../log/stocks/all/evaluate_sssd_all_s4_seq_800.log  2>&1 &

#########################################################################################################################

nohup python3 -u train-sssd.py --cuda 0 --alg "S5" --stock 'all' --batch_size 8 --seq_len 200 --num_layers 18 > ../log/stocks/all/train_sssd_all_s5_seq_200.log  2>&1 &
nohup python3 -u train-sssd.py --cuda 1 --alg "Mega" --stock 'all' --batch_size 8 --seq_len 200  --num_layers 18 > ../log/stocks/all/train_sssd_all_mega_seq_200.log  2>&1 &

nohup python3 -u train_imputers.py --cuda 0 --model 'csdi' --alg 'Mega' --seq_len 200 --batch_size 32 > ../log/stocks/all/train_csdi_mega_all_seq_200.log  2>&1 &

nohup python3 -u train-csdi.py --cuda 0 --alg "S5" --stock 'all' --batch_size 32 --seq_len 200 > ../log/stocks/all/train_csdi_all_s5_seq_200.log  2>&1 &
nohup python3 -u train-csdi.py --cuda 0 --alg "Mega" --stock 'all' --batch_size 32 --seq_len 200 > ../log/stocks/all/train_csdi_all_mega_seq_200.log  2>&1 &
nohup python3 -u train-csdi.py --cuda 0 --alg "transformer" --stock 'all' --batch_size 64 --seq_len 200 > ../log/stocks/all/train_csdi_all_transofrmer_seq_200.log  2>&1 &

nohup python3 -u train_imputers.py --cuda 0 --model 'mega' --seq_len 100 --batch_size 16 > ../log/stocks/all/train_mega_all_seq_100.log  2>&1 &
nohup python3 -u train_imputers.py --cuda 0 --model 'mega' --seq_len 200 --batch_size 16 > ../log/stocks/all/train_mega_all_seq_200.log  2>&1 &
nohup python3 -u train_imputers.py --cuda 0 --model 'mega' --seq_len 400 --batch_size 8 > ../log/stocks/all/train_mega_all_seq_400.log  2>&1 &
nohup python3 -u train_imputers.py --cuda 0 --model 'mega' --seq_len 800 --batch_size 8 > ../log/stocks/all/train_mega_all_seq_800.log  2>&1 &


nohup python3 -u evaluate_mega.py --cuda 0  --stock 'all' --batch_size 128 --seq_len 200 > ../log/stocks/all/evaluate_mega_all_seq_200.log  2>&1 &
nohup python3 -u evaluate_mega.py --cuda 0  --stock 'all' --batch_size 64 --seq_len 400 > ../log/stocks/all/evaluate_mega_all_seq_400.log  2>&1 &
nohup python3 -u evaluate_mega.py --cuda 0  --stock 'all' --batch_size 64 --seq_len 800  > ../log/stocks/all/evaluate_mega_all_seq_800.log  2>&1 &





nohup python3 -u evaluate_csdi.py --cuda 0 --alg "transformer" --stock 'all' --batch_size 16 --seq_len 200 > ../log/stocks/all/evaluate_csdi_all_transformer_seq_200.log  2>&1 &




########################################################################################################################
nohup python3 -u evaluate_sssd.py --algo 'S4' --cuda 0 --stock 'all' > ../log/stocks/all/evaluate_sssd_all_s4.log  2>&1 &

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


nohup python3 -u predict.py > ../log/prediction_task/predict_hk_stocks.log  2>&1 &

