from transformers import AutoTokenizer, AutoModelWithLMHead
from custom_dataset import WoZDSTDataset
from typing import Dict, Optional
import os
import sklearn


import argparse
train_arg_parser = argparse.ArgumentParser(description="CoPINpet")
train_arg_parser.add_argument("--local_rank", default=0, type=int)
train_arg_parser.add_argument("--model-type", default='distilbert-base-cased', type=str)
train_arg_parser.add_argument("--restore", default='', type=str)
train_arg_parser.add_argument("--seed", default=1234, type=int)
train_arg_parser.add_argument("--max-length", default=512, type=int)
train_arg_parser.add_argument("--option", default='base', type=str)
train_arg_parser.add_argument("--is-debug", action='store_true')
train_arg_parser.add_argument("--mlm", type=float, default=0.15)
train_arg_parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
train_arg_parser.add_argument("--batch-size", default=32, type=int)
train_arg_parser.add_argument("--num-train-epochs", default=50, type=int)
train_arg_parser.add_argument("--pretrain-gate", action='store_true')
train_arg_parser.add_argument("--pretrain-uttembed", action='store_true')
train_arg_parser.add_argument("--data-option", type=str)
train_arg_parser.add_argument("--fp-16",action='store_true')
train_arg_parser.add_argument("--version",type=str, default='1')

args = train_arg_parser.parse_args()

default_batch_size= 16
# if args.batch_size < 16:
#     gradient_accumulation_steps = default_batch_size / args.batch_size
# else:
# gradient_accumulation_steps = 1

args.output_dir = os.path.join('result',
                               'pre_'
                               f'{args.model_type}_'                               
                               f'b-{args.batch_size}_'
                               f'g-{args.gradient_accumulation_steps}_'
                               f'd-{args.max_length}_'
                               f'm-{args.mlm}_'
                               f'o-{args.option}_'
                               f'd-{args.data_option}_'
                               f'pg-{args.pretrain_gate}_'
                               f'u-{args.pretrain_uttembed}_'
                               f'f16-{args.fp_16}_'
                               f'v-{args.version}_'
                               f'{args.seed}'
                         )
if args.restore:
    args.output_dir = args.output_dir+'_restored'

if args.is_debug:
    args.output_dir = args.output_dir + '_test'



from my_tokenization_albert import AlbertTokenizer
# tokenizer = AutoTokenizer.from_pretrained(args.model_type)
tokenizer = AlbertTokenizer.from_pretrained(args.model_type)


# domain_special_tokens = [f'[D_{i}]' for i in range(5)] + [f'[S_{i}]' for i in range(17)]
domain_special_tokens =[f'[D_{i}]' for i in range(30)]
tokenizer.add_special_tokens({'additional_special_tokens': ['[SEPT]'] + domain_special_tokens})

print('ALBERT WITH UTTERANCE TYPE EMBED' * 10)
from my_modeling_albert import AlbertForPreTraining as myAlbertForPreTraining
# model = myAlbertForPreTraining.from_pretrained(args.model_type, num_labels=args.dialog_turn_length)

from transformers import AutoConfig

if args.restore:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.restore)
    config.version = args.version
    config.num_labels = 50
    model = myAlbertForPreTraining.from_pretrained(args.restore, config=config)
else:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model_type)
    config.version = args.version
    config.num_labels = 50 # max 50
    model = myAlbertForPreTraining.from_pretrained(args.model_type, config=config)


# model.config.num_labels = args.dialog_turn_length
model.config.train_mode = 'pretrain'
model.config.restore_dir = args.restore
model.config.domain_cls_tokens = tokenizer.additional_special_tokens_ids[1:]
model.config.sept_token_id = 30000

# from my_utils_multiWOZ_DST import prepare_data_seq
# train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = \
#     prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']))

is_pretrain=True
train_dataset = WoZDSTDataset(tokenizer, 'train', args.max_length, args.data_option, is_pretrain,
                              args.pretrain_gate, args.pretrain_uttembed)
test_dataset = WoZDSTDataset(tokenizer, 'test', args.max_length, args.data_option, is_pretrain,
                             args.pretrain_gate, args.pretrain_uttembed)
model.resize_token_embeddings(len(tokenizer))

if args.max_length > 512:
    max_position_embeddings = 1024
    model.resize_positional_embeddings(max_position_embeddings)
    # model.config.vocab_size = max_position_embeddings

# from transformers import DataCollatorForLanguageModeling
#
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer, mlm=True, mlm_probability=0.15
# )

from my_data_collator import MyDataCollatorForLanguageModeling

data_collator = MyDataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm
)


from transformers import TrainingArguments

from my_trainer import Trainer

# from transformers.trainer_utils import EvalPrediction
from my_trainer_utils import EvalPrediction

# class MyEvalPrediction(EvalPrediction):
#
#     def __call__(self, predictions, label_ids):
#
if args.is_debug:
    eval_steps = 1
    save_steps = 1
    logging_steps = 1
    fp_16=args.fp_16
else:
    eval_steps = 8_000
    save_steps = 5_000
    logging_steps = 1_000
    fp_16 = args.fp_16



training_args = TrainingArguments(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    save_steps=save_steps,
    save_total_limit=None,
    fp16=args.fp_16,
    fp16_opt_level='O1',
    logging_steps=logging_steps,
    logging_dir=args.output_dir,
    local_rank=args.local_rank,
    evaluate_during_training=True,
    do_eval=True,
    eval_steps=eval_steps,
    logging_first_step=True,
    seed = args.seed
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


def compute_metrics_mlm(p: EvalPrediction) -> Dict:
    report_dict =dict()
    if p.domain_predictions is not None:
        domain_accuracy = (p.domain_predictions == p.domain_label_ids).sum() / p.domain_label_ids.size
        report_dict['domain_accuracy'] = domain_accuracy
    if p.gate_predictions is not None:
        gate_accuracy = (p.gate_predictions == p.gate_label_ids).sum() / p.gate_label_ids.size

        joint_gate_accuracy = (p.gate_predictions == p.gate_label_ids).all(1).sum() / p.gate_label_ids.shape[0]
        # gate_accuracy = (p.sop_predictions == p.sentence_order_label_ids).sum() / p.sentence_order_label_ids.size
        # gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
        gating_dict = {0: "ptr", 1: "dontcare", 2: "none"}
        gate_prec = sklearn.metrics.precision_score(p.gate_label_ids.flatten(), p.gate_predictions.flatten(), average=None)
        gate_recall = sklearn.metrics.recall_score(p.gate_label_ids.flatten(), p.gate_predictions.flatten(), average=None)
        report_dict.update({"joint_gate_accuracy": joint_gate_accuracy,
                            'gate_accuracy': gate_accuracy})
        for idx, val in enumerate(gate_prec):
            report_dict[f'prec_{gating_dict[idx]}'] = val
        for idx, val in enumerate(gate_recall):
            report_dict[f'rec_{gating_dict[idx]}'] = val

    mlm_accuracy = (p.predictions == p.label_ids).sum() / p.label_ids.size
    report_dict.update({"mlm_accuracy": mlm_accuracy})
    if p.sop_predictions is not None:
        sop_accuracy = (p.sop_predictions == p.sentence_order_label_ids).sum() / p.sentence_order_label_ids.size
        report_dict.update({"sop_accuracy": sop_accuracy})
    # sop_f1 = f1_score(p.sentence_order_label_ids, p.sop_predictions, pos_label=0)
    # sop_prec = precision_score(p.sentence_order_label_ids, p.sop_predictions, pos_label=0)
    # sop_recall = recall_score(p.sentence_order_label_ids, p.sop_predictions, pos_label=0)
    # success = p.predictions.tolist() == sequential and p.label_ids.tolist() == sequential



    return report_dict

trainer = Trainer(
    model=model,
    args=training_args,
    model_config = vars(args),
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    prediction_loss_only=False,
    compute_metrics=compute_metrics_mlm,
    metrics_name_list=['mlm_accuracy', 'sop_accuracy']
)

trainer.train()
# trainer.evaluate()