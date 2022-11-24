import pickle

from typing import Dict
import os
import sklearn
import re
import time
import copy
import argparse
import torch

train_arg_parser = argparse.ArgumentParser(description="WOZ")
train_arg_parser.add_argument("--local_rank", default=0, type=int)
train_arg_parser.add_argument("--model-type", default='albert-base-v2', type=str)
train_arg_parser.add_argument("--restore", default='', type=str)
train_arg_parser.add_argument("--restore-finetune", default='', type=str)
train_arg_parser.add_argument("--seed", default=1234, type=int)
train_arg_parser.add_argument("--max-length", default=512, type=int)
train_arg_parser.add_argument("--option", default='base', type=str)
train_arg_parser.add_argument("--ds-type", default='split', type=str)
train_arg_parser.add_argument("--is-debug", action='store_true')
train_arg_parser.add_argument("--batch-size", default=32, type=int)
train_arg_parser.add_argument("--learning-rate", default=5e-5, type=float)
train_arg_parser.add_argument("--warmup-steps", default=1000, type=int)
train_arg_parser.add_argument("--num-train-epochs", default=70, type=int)
train_arg_parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
train_arg_parser.add_argument("--no-utt-embed", action='store_true')
train_arg_parser.add_argument("--max-turn", default=4, type=int)
train_arg_parser.add_argument("--fp-16",action='store_true')
train_arg_parser.add_argument("--no-d-attn",action='store_true')
train_arg_parser.add_argument("--delta-ds",action='store_true')
train_arg_parser.add_argument("--order_shuffle",action='store_true')
train_arg_parser.add_argument("--sln",type=float)

train_arg_parser.add_argument("--data-option", type=str)
train_arg_parser.add_argument("--version", type=str, default='1')

args = train_arg_parser.parse_args()

args.output_dir = os.path.join('pretrain_result',
                               f'{args.model_type}_'                               
                               f'b-{args.batch_size}_'
                               f'g-{args.gradient_accumulation_steps}_'                               
                               f'd-{args.data_option}_'                               
                               f'ds-{args.ds_type}_'
                               f'w-{args.warmup_steps}_'
                               f'lr-{args.learning_rate}_'
                               f'tr-{args.num_train_epochs}_'
                               f'mt-{args.max_turn}_'                               
                               f'o-{args.option}_'
                               f'ms-{args.delta_ds}_'
                               f'os-{args.order_shuffle}_'
                               f'sln-{args.sln}_'
                               f'{args.seed}'
                               )
if args.restore:
    args.restore =os.path.join('../result', args.restore)
    args.output_dir = args.output_dir+'_restored'
    best_restored = sorted([x for x in os.listdir(args.restore) if re.search('best-model-(\d+)', x)],
                           key = lambda x: int(re.search('best-model-(\d+)', x).group(1)), reverse=True)
    args.restore = os.path.join(args.restore, best_restored[0])
    print('restore from', args.restore)

if args.restore_finetune:
    args.restore_finetune =os.path.join('../result', args.restore_finetune)
    args.output_dir = args.output_dir+'_restore_finetune'
    # best_restored = sorted([x for x in os.listdir(args.restore_finetune) if re.search('best-acc-model-(\d+)', x)],
    #                        key = lambda x: int(re.search('best-acc-model-(\d+)', x).group(1)), reverse=True)
    # args.restore_finetune = os.path.join(args.restore_finetune, best_restored[0])
    # print('candidate checkpoint dir', os.listdir(args.restore_finetune))
    # restore_dir = input('which to choose?')

    best_restored = sorted([x for x in os.listdir(args.restore_finetune) if re.search('interval-(\d+)', x)],
                           key=lambda x: int(re.search('interval-(\d+)', x).group(1)), reverse=True)
    args.restore_finetune = os.path.join(args.restore_finetune, best_restored[0])
    print('checkpoint final', os.listdir(args.restore_finetune))
    print('restore finetune from', args.restore_finetune)

if args.is_debug:
    args.output_dir = args.output_dir + '_test'

if 'albert' in args.model_type:
    from transformers import AlbertTokenizer

    class MyTokenizer(AlbertTokenizer):
        def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
            if already_has_special_tokens:
                if token_ids_1 is not None:
                    raise ValueError(
                        "You should not supply a second sequence if the provided sequence of "
                        "ids is already formatted with special tokens for the model."
                    )
                return list(map(
                    lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] + self.additional_special_tokens_ids else 0,
                    token_ids_0))

    tokenizer = MyTokenizer.from_pretrained(args.model_type)
elif 'robert' in args.model_type:
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_type)


domain_special_tokens = [f'[DS_{i}]' for i in range(30)]
# elif args.ds_type == 'none':
#     domain_special_tokens = []
other_special_tokens = ['[USR]', '[SYS]', '[SEP_DS]']
tokenizer.add_special_tokens({'additional_special_tokens': other_special_tokens + domain_special_tokens})
tokenizer.usr_token = '[USR]'
tokenizer.sys_token = '[SYS]'
tokenizer.sep_ds_token = '[SEP_DS]'
tokenizer.domain_special_tokens = domain_special_tokens


is_pretrain = True

if args.version == '1':
    from custom_dataset_autoregressive import WoZDSTDataset

    train_dataset = WoZDSTDataset(tokenizer, 'train', args.max_length, args.data_option, is_pretrain, args.max_turn, args.is_debug, args.delta_ds,args.order_shuffle)
    test_dataset = WoZDSTDataset(tokenizer, 'test', args.max_length, args.data_option, is_pretrain, args.max_turn, args.is_debug, args.delta_ds, args.order_shuffle)
    validation_dataset = WoZDSTDataset(tokenizer, 'dev', args.max_length, args.data_option, is_pretrain, args.max_turn, args.is_debug, args.delta_ds, args.order_shuffle)
elif args.version == '2':
    train_dataset = WoZDialogDataset(tokenizer, 'train', args.max_length, args.data_option,
                                  is_pretrain)
    test_dataset = WoZDialogDataset(tokenizer, 'test', args.max_length, args.data_option,
                                 is_pretrain)




# model = AutoModelWithLMHead.from_pretrained(args.model_type)
# from transformers.modeling_albert import AlbertForPreTraining
# model = AlbertForPreTraining.from_pretrained(args.model_type)

time.sleep(2) # waiting for the data/trade_slot_value_dict.pkl to be created

from my_modeling_albert import AlbertForPreTraining as myAlbertForPreTraining

# if 'albert' in args.model_type:
#     from my_dst_modeling_albert import AlbertForDST as modelforDST
# elif 'robert' in args.model_type:
#     from my_dst_modeling_roberta import RobertaForDST as modelforDST


from transformers import AutoConfig


if args.restore:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.restore)
    config.version = args.version
    model = myAlbertForPreTraining.from_pretrained(args.restore, config=config)
elif args.restore_finetune:
    print('RESTORING FINETUNE', args.restore_finetune)
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.restore_finetune)
    config.version = args.version
    config.domain_special_tokens = tokenizer.domain_special_tokens
    config.ds_type = args.ds_type
    config.data_option = args.data_option
    config.is_domain_attention = not args.no_d_attn
    model = myAlbertForPreTraining.from_pretrained(args.restore_finetune, config=config)
else:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model_type)
    config.version = args.version
    config.domain_special_tokens = tokenizer.domain_special_tokens
    config.ds_type = args.ds_type
    config.data_option = args.data_option
    config.is_domain_attention = not args.no_d_attn
    model = myAlbertForPreTraining.from_pretrained(args.model_type, config=config)

    if args.max_length > 512:
        max_position_embeddings = args.max_length
        model.resize_positional_embeddings(max_position_embeddings)
        config.max_position_embeddings = max_position_embeddings

    model.resize_token_embeddings(len(tokenizer))


model.config.train_mode = 'dst'
model.config.restore_dir = args.restore
model.config.restore_finetune_dir = args.restore_finetune
model.config.domain_cls_tokens = tokenizer.domain_special_tokens
model.config.sln = args.sln




# from my_utils_multiWOZ_DST import prepare_data_seq
# train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = \
#     prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']))




# from transformers import DataCollatorForLanguageModeling
#
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
# )

from my_data_collator import MyDataCollatorForLanguageModeling

data_collator = MyDataCollatorForLanguageModeling(
    tokenizer=tokenizer
)


from transformers import TrainingArguments

# from my_dst_trainer import Trainer
from my_trainer import Trainer

# from transformers.trainer_utils import EvalPrediction
from my_trainer_utils import DSTEvalPrediction
from my_trainer_utils import EvalPrediction

#
#     def __call__(self, predictions, label_ids):

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
else:
    device_count = 1
if args.is_debug:
    eval_steps = 100
    save_steps = 100
    logging_steps = 1
    fp_16=False
else:
    eval_steps = int(4000 // ((args.gradient_accumulation_steps * args.batch_size * device_count) / 32))
    save_steps = eval_steps
    logging_steps = int(500 // ((args.gradient_accumulation_steps * args.batch_size * device_count) / 32))
    fp_16 = args.fp_16

training_args = TrainingArguments(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=2,
    save_steps=save_steps,
    save_total_limit=None,
    fp16=fp_16,
    fp16_opt_level='O1',
    logging_steps=logging_steps,
    logging_dir=args.output_dir,
    local_rank=args.local_rank,
    evaluate_during_training=True,
    do_eval=True,
    eval_steps=eval_steps,
    logging_first_step=True, #This acutaully means log early ( after 10 steps)
    seed = args.seed,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_steps
)

# import logging
# logger = logging.getLogger(__name__)
# logger.warning(
#     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
#     training_args.local_rank,
#     training_args.device,
#     training_args.n_gpu,
#     training_args.local_rank != -1,
# )

def compute_metrics_pretrain(p: EvalPrediction) -> Dict:
    # sequential = list(range(len(dataset)))
    domain_slot_list = train_dataset.domain_slot_list


    report_dict = dict()

    mlm_mask = p.label_ids != -100
    predictions_mlm = p.predictions[mlm_mask]
    label_mlm = p.label_ids[mlm_mask]

    mlm_accuracy = (predictions_mlm == label_mlm).sum() / label_mlm.size

    report_dict.update({"mlm_accuracy" : mlm_accuracy})
    if args.delta_ds:
        modification_f1 = sklearn.metrics.f1_score(p.modified_slot_label.flatten(), p.modified_preds.flatten())
        modification_rec = sklearn.metrics.recall_score(p.modified_slot_label.flatten(), p.modified_preds.flatten())
        modification_prec = sklearn.metrics.precision_score(p.modified_slot_label.flatten(), p.modified_preds.flatten())

        report_dict.update({"modification_f1": modification_f1,
                            "modification_rec": modification_rec,
                            "modification_prec": modification_prec})

    if args.order_shuffle:
        in_order_accuracy = sklearn.metrics.accuracy_score(p.in_order_label_ids.flatten(), p.in_order_preds.flatten())

        report_dict.update({"in_order_accuracy": in_order_accuracy})



    return report_dict

trainer = Trainer(
    model=model,
    args=training_args,
    model_config = vars(args),
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    validation_dataset=validation_dataset,
    prediction_loss_only=False,
    compute_metrics=compute_metrics_pretrain,
    metrics_name_list=['joint_gate_accuracy', 'gate_accuracy']
)

if args.restore_finetune:
    model_path = args.restore_finetune
else:
    model_path = None
trainer.train(model_path=model_path)

# trainer.evaluate()