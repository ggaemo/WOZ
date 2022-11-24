#python -m torch.distributed.launch --nproc_per_node=8 src/main_pretrain.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option nepl_revision_hyperparam_1 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 2

python -m torch.distributed.launch --nproc_per_node=8 src/main_pretrain.py --model-type albert-base-v2 --data-option with_ds_special_wo_none_same_domain --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option nepl_revision_hyperparam_2_mlm_shuffle_same_domain --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5 --delta-ds --order_shuffle

#python -m torch.distributed.launch --nproc_per_node=8 src/main_pretrain.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option nepl_revision_hyperparam_3 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 2 --delta-ds

#python -m torch.distributed.launch --nproc_per_node=8 src/main_pretrain.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option nepl_revision_hyperparam_4 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 2 --order_shuffle




#python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option final_every10_msos_real_shuffle_2 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5  --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-pilot_shuffle_every10_ms-True_os-True_1234  --pret_idx 0

#python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option final_every10_msos_real_shuffle_2 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5  --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-pilot_shuffle_every10_ms-True_os-True_1234  --pret_idx 10

#python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option final_every10_msos_real_shuffle_2 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5  --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-pilot_shuffle_every10_ms-True_os-True_1234  --pret_idx 5

#python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option final_every10_msos_real_shuffle_2 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5  --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-pilot_shuffle_every10_ms-True_os-True_1234  --pret_idx 2

#python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option final_every10_msos_real_shuffle_2 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5  --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-pilot_shuffle_every10_ms-True_os-True_1234  --pret_idx 8

#python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option final_every10_msos_real_shuffle_2 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5  --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-pilot_shuffle_every10_ms-True_os-True_1234  --pret_idx 15

#python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option final_every10_msos_real_shuffle_2 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5  --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-pilot_shuffle_every10_ms-True_os-True_1234  --pret_idx 19




#python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 16 --max-length 512 --gradient-accumulation-steps 1 --option emnlp_shuffle_every10_msos_retry --num-train-epochs 100 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5  --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-pilot_shuffle_every10_ms-True_os-False_1234 
