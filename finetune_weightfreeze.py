from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, BertTokenizer, RobertaTokenizer
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
from datasets import Dataset, load_dataset
from dataloaders import *
import argparse
import pickle
import os
import wandb
from transformers import set_seed


def init_weights_with_seed(m, init_seed=3005):
    # Save the current state of random number generators
    print("Init wieghts ...")
    rng_state = torch.get_rng_state()
    print(torch.get_rng_state())

    # Set the seed specifically for initialization
    torch.manual_seed(init_seed)
    print(torch.get_rng_state())

    if isinstance(m, nn.Linear):
        # Example of using Xavier Uniform Initialization
        init.xavier_uniform_(m.weight)  # Xavier initialization
        if m.bias is not None:
            init.constant_(m.bias, 0.0)  # Bias initialization to 0

    # Restore the RNG state to prevent affecting other randomness
    print(torch.get_rng_state())

    torch.set_rng_state(rng_state)
    print(torch.get_rng_state())


# set_seed(3005)

parser = argparse.ArgumentParser()

parser.add_argument("--task", required=True,
                    help="Tasks like ClimaText, SciDCC")
parser.add_argument("--run-name", required=True,
                    help="Run Name for wandb logging")
parser.add_argument("--model", type=str, default="bert-base-uncased",
                    help="huggingface model to train and test")
parser.add_argument("--max-len", type=int, default=512,
                    help="huggingface model max length")
parser.add_argument("--epochs", default=5, type=int,
                    help="number of epochs to train")
parser.add_argument("--per_device_train_batch_size", default=32, type=int,
                    help="per_device_train_batch_size")
parser.add_argument("--per_device_eval_batch_size", default=64, type=int,
                    help="per_device_eval_batch_size")
parser.add_argument("--seed", type=int, default=0,  # Default seed is 42
                    help="Random seed for reproducibility")
args = parser.parse_args()

set_seed(args.seed) # original seed 0

class CustomTrainer(Trainer):

    def __init__(self, **inputs):
        self.class_weights = inputs.pop("class_weights")
        Trainer.__init__(self, **inputs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        if self.class_weights is None:
            weight = None
        else:
            weight = torch.Tensor([self.class_weights] if type(self.class_weights) == np.float64 else self.class_weights).cuda()

        if self.model.config.num_labels == 1:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=weight)
            labels = labels.float()
            logits = logits.view(-1)
        else:
            loss_fct = nn.CrossEntropyLoss(weight)
            logits = logits.view(-1, self.model.config.num_labels)

        loss = loss_fct(logits, labels.view(-1))
        return (loss, outputs) if return_outputs else loss


os.environ["WANDB_PROJECT"] = 'climate_glue_acl_weightfreeze'
os.environ["WANDB_RUN_GROUP"] = args.task


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
class_weights = None
if args.task == 'ClimaText':
    files = {'train': ['ClimaText/train-data/AL-10Ks.tsv : 3000 (58 positives, 2942 negatives) (TSV, 127138 KB).tsv',
                       'ClimaText/train-data/AL-Wiki (train).tsv'],
             'val': ['ClimaText/dev-data/Wikipedia (dev).tsv'],
             'test': ['ClimaText/test-data/Claims (test).tsv',
                      'ClimaText/test-data/Wikipedia (test).tsv',
                      'ClimaText/test-data/10-Ks (2018, test).tsv']
             }
    data_class = ClimaText(files)
elif args.task == 'SciDCC':
    file = 'SciDCC/SciDCC.csv'
    data_class = SciDCC(file)

elif args.task == 'CDPCities':
    folder = './CDP/Cities/Cities Responses/'
    data_class = CDPCities(folder)
elif args.task == 'ClimateStance':
    files = {'train': 'ClimateStance/train.csv',
             'val': 'ClimateStance/val.csv',
             'test': 'ClimateStance/test.csv'
             }
    data_class = ClimateStance(files)
elif args.task == 'ClimateEng':
    files = {'train': 'ClimateEng/train.csv',
             'val': 'ClimateEng/val.csv',
             'test': 'ClimateEng/test.csv'
             }
    data_class = ClimateEng(files)
elif args.task == "ClimateInsurance":
    files = {'train': 'ClimateInsurance/train.csv',
             'val': 'ClimateInsurance/val.csv',
             'test': 'ClimateInsurance/test.csv'
             }
    data_class = ClimateInsurance(files)
elif args.task == "ClimateInsuranceMulti":
    files = {'train': 'ClimateInsuranceMulti/train.csv',
             'val': 'ClimateInsuranceMulti/val.csv',
             'test': 'ClimateInsuranceMulti/test.csv'
             }
    data_class = ClimateInsuranceMulti(files)
elif args.task == 'CDPCitiesQA':
    folder = './CDP/Cities/Cities Responses/'
    data_class = CDPQA(folder)
elif args.task == 'CDPStatesQA':
    folder = './CDP/States/'
    data_class = CDPQA(folder)
elif args.task == 'CDPCorpsQA':
    folder = './CDP/Corporations/Corporations Responses/Climate Change/'
    data_class = CDPQA(folder)
elif args.task == 'CDPCombinedQA':
    folder = './CDP/Combined/'
    data_class = CDPQA(folder)
elif args.task == 'ClimateFEVER':
    data_class = ClimateFEVER()


def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
    assert len(scores) == len(labels)
    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    max_acc = 0
    best_threshold = -1

    positive_so_far = 0
    remaining_negatives = sum(labels == 0)

    for i in range(len(rows)-1):
        score, label = rows[i]
        if label == 1:
            positive_so_far += 1
        else:
            remaining_negatives -= 1

        acc = (positive_so_far + remaining_negatives) / len(labels)
        if acc > max_acc:
            max_acc = acc
            best_threshold = (rows[i][0] + rows[i+1][0]) / 2

    return max_acc, best_threshold[0]


def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    best_f1 = best_precision = best_recall = 0
    threshold = 0
    nextract = 0
    ncorrect = 0
    total_num_duplicates = sum(labels)

    for i in range(len(rows)-1):
        score, label = rows[i]
        nextract += 1

        if label == 1:
            ncorrect += 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_num_duplicates
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return best_f1, best_precision, best_recall, threshold[0]


def compute_metrics(pred):
    labels = pred.label_ids
    if data_class.num_labels == 1:
        preds = pred.predictions
        acc, acc_threshold = find_best_acc_and_threshold(preds, labels, True)
        f1, precision, recall, f1_threshold = find_best_f1_and_threshold(preds, labels, True)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'f1_threshold': f1_threshold,
            'acc_threshold': acc_threshold
        }
    preds = pred.predictions.argmax(-1)
    precision, recall, f1_macro, support = precision_recall_fscore_support(labels, preds, average='macro')
    f1_micro = f1_score(labels, preds, average='weighted')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'support': support
    }


if "CliReBERT" in args.model:
    print("Loading (CliReBERT): ", args.model)
    tokenizer = BertTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
if "CliSciBERT" in args.model:
    print("Loading (CliSciBERT): ", args.model)
    tokenizer = BertTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
if "SciClimateBERT" in args.model:
    print("Loading (SciClimateBERT): ", args.model)
    tokenizer = RobertaTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
if "CliReRoBERTa" in args.model:
    print("Loading (CliReRoBERTa): ", args.model)
    tokenizer = RobertaTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model,
                                                           num_labels=data_class.num_labels)

print(torch.get_rng_state())
# Make weights of the head always same
model.apply(init_weights_with_seed)
# for name, param in model.named_parameters():
#     print(name, param)
print(torch.get_rng_state())
train_dataset, val_dataset, test_dataset = data_class.prepare(tokenizer, args.max_len)

training_args = TrainingArguments(
    output_dir=f'/projects/user/climate_glue_acl_wf/results/{args.task}/{args.model}_{args.seed}',
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='/projects/user/climate_glue_acl_wf/logs',
    dataloader_num_workers=8,
    report_to="wandb",
    run_name=args.run_name,
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1_macro',
    greater_is_better=True,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    fp16=True,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    class_weights=data_class.class_weights,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
trainer.save_model()
predictions, label_ids, metrics = trainer.predict(val_dataset)
print(f'Metrics for val set: {metrics}')
predictions, label_ids, metrics = trainer.predict(test_dataset)
result = {args.task: {'predictions': predictions, 'label_ids': label_ids, 'metrics': metrics}}

with open(f'test_results/wf_result_{args.task}_{args.model.replace("/","_")}_{args.seed}.pickle', 'wb') as f:
    pickle.dump(result, f)

print(metrics)
print(data_class.labels)
wandb.log(metrics)

