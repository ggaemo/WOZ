python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 4 --max-length 512 --gradient-accumulation-steps 2 --option nepl_last_epoch_restore_ms-T_os-T_same_domain --pret_idx 0 --num-train-epochs 50 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5 --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_same_domain_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-nepl_revision_hyperparam_2_mlm_shuffle_same_domain_ms-True_os-True_sln-None_1234 
sleep 5s
python -m torch.distributed.launch --nproc_per_node=8 src/main_dst.py --model-type albert-base-v2 --data-option with_ds_special_wo_none --batch-size 4 --max-length 512 --gradient-accumulation-steps 2 --option nepl_last_epoch_restore_ms-T_os-T_same_domain --pret_idx 0 --num-train-epochs 100 --warmup-steps 100 --ds-type merged --learning-rate 5e-5 --fp-16 --max-turn 5 --restore_pretrain albert-base-v2_b-16_g-1_d-with_ds_special_wo_none_same_domain_ds-merged_w-100_lr-5e-05_tr-50_mt-5_o-nepl_revision_hyperparam_2_mlm_shuffle_same_domain_ms-True_os-True_sln-None_1234 








