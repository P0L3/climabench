from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, BertTokenizer, RobertaTokenizer
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
from datasets import Dataset, load_dataset, concatenate_datasets
from dataloaders import *
import argparse
import pickle
import os
import wandb
from transformers import set_seed
from sklearn.model_selection import KFold, StratifiedKFold



# set_seed(3005)

parser = argparse.ArgumentParser()

# In your argument parsing section (e.g., using argparse)
parser.add_argument(
    "--k_folds",
    type=int,
    default=None,
    help="Number of folds for K-Fold cross-validation. If set, train/validation split args are ignored for splitting, but used for loading initial data."
)
parser.add_argument(
    "--k_fold_shuffle",
    action="store_true",
    help="Shuffle data before splitting into folds. Recommended."
)
parser.add_argument(
    "--use_stratified_kfold",
    action="store_true",
    help="Use StratifiedKFold instead of KFold. Requires label_column_name."
)
# Make sure you have an argument for the label column name if using StratifiedKFold
parser.add_argument(
    "--label_column_name",
    type=str,
    default="label", # Or your default label column
    help="The name of the column containing the labels, needed for stratified k-fold."
)

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


os.environ["WANDB_PROJECT"] = 'climate_glue_acl_Kfold'
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


## Load the tokenizer
if "CliReBERT" in args.model:
    print("Loading (CliReBERT) tokenizer: ", args.model)
    tokenizer = BertTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
if "CliSciBERT" in args.model:
    print("Loading (CliSciBERT) tokenizer: ", args.model)
    tokenizer = BertTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
if "SciClimateBERT" in args.model:
    print("Loading (SciClimateBERT) tokenizer: ", args.model)
    tokenizer = RobertaTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
if "CliReRoBERTa" in args.model:
    print("Loading (CliReRoBERTa) tokenizer: ", args.model)
    tokenizer = RobertaTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)



## Prepare and load full dataset
train_dataset, val_dataset, test_dataset = data_class.prepare(tokenizer)



if args.k_folds is not None and args.k_folds > 1:
    print(f" K-Fold Cross-Validation enabled with k={args.k_folds}")

    # 1. Merge Datasets if both train and validation were provided
    print("Merging train and validation datasets for K-Fold splitting.")
        # Check column consistency if needed
    if set(train_dataset.column_names) != set(val_dataset.column_names) or set(train_dataset.column_names) != set(test_dataset.column_names):
        # Implement logic to align columns (e.g., keep common, rename)
        # This is a placeholder - adapt to your needs
        print("Warning: Train, test and validation dataset columns differ. Ensure alignment.")
        common_columns = list(set(train_dataset.column_names) & set(val_dataset.column_names) & set(test_dataset.column_names))

        train_dataset = train_dataset.select_columns(common_columns)
        val_dataset = val_dataset.select_columns(common_columns)
        test_dataset = test_dataset.select_columns(common_columns)
        


    full_dataset = concatenate_datasets([train_dataset, val_dataset, test_dataset])


    # Ensure the dataset is shuffled if requested (KFold/StratifiedKFold might do it too,
    # but doing it explicitly on the dataset object can be good practice).
    if args.k_fold_shuffle:
         full_dataset = full_dataset.shuffle(seed=args.seed)
         
    # 2. Prepare for Splitting
    if args.use_stratified_kfold:
        if args.label_column_name not in full_dataset.column_names:
            raise ValueError(f"Label column '{args.label_column_name}' not found in dataset for StratifiedKFold.")
        labels = full_dataset[args.label_column_name]
        kf = StratifiedKFold(n_splits=args.k_folds, shuffle=args.k_fold_shuffle, random_state=args.seed)
        split_gen = kf.split(np.arange(len(full_dataset)), labels) # Use labels for stratification
        print("Using Stratified K-Fold.")
    else:
        kf = KFold(n_splits=args.k_folds, shuffle=args.k_fold_shuffle, random_state=args.seed)
        split_gen = kf.split(np.arange(len(full_dataset))) # Just split based on indices
        print("Using standard K-Fold.")

    all_fold_metrics = [] # To store metrics from each fold ####DO TU
    # 3. Loop through Folds
    for fold_idx, (train_indices, test_indices) in enumerate(split_gen):
        print(f"\n--- Starting Fold {fold_idx + 1}/{args.k_folds} ---")

        # **CRITICAL: Re-initialize model from scratch for each fold**
        # This prevents knowledge leakage from previous folds. 
        if "CliReBERT" in args.model:
            print("Loading (CliReBERT): ", args.model)
            # tokenizer = BertTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
        if "CliSciBERT" in args.model:
            print("Loading (CliSciBERT): ", args.model)
            # tokenizer = BertTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
        if "SciClimateBERT" in args.model:
            print("Loading (SciClimateBERT): ", args.model)
            # tokenizer = RobertaTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
        if "CliReRoBERTa" in args.model:
            print("Loading (CliReRoBERTa): ", args.model)
            # tokenizer = RobertaTokenizer.from_pretrained(args.model)#pretrain_path, vocab_file=pretrain_path + "/tokenizer.json")
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)
        else:
            # tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=data_class.num_labels)


        # Create train/validation datasets for the current fold
        train_fold_dataset_full = full_dataset.select(train_indices)
        
        # --- Split train_fold_dataset_full into train/validation for the Trainer ---
        print(f"Splitting fold {fold_idx + 1} training data into inner train/validation sets...")
        try:
            # Use train_test_split from the datasets library for easy splitting
            # Stratify this inner split if possible and requested
            stratify_column = args.label_column_name if args.use_stratified_kfold else None
            train_val_split = train_fold_dataset_full.train_test_split(
                test_size=args.validation_split_percentage,
                seed=args.seed + fold_idx, # Use a seed, vary slightly per fold
                stratify_by_column=stratify_column
            )
            # These will be used by the Trainer
            train_fold_dataset = train_val_split['train']
            val_fold_dataset = train_val_split['test']
            print(f"Inner split: {len(train_fold_dataset)} train, {len(val_fold_dataset)} validation samples.")

        except ValueError as e:
             # Handle cases where stratification might fail (e.g., too few samples for a class)
             print(f"Warning: Stratified train/validation split failed for fold {fold_idx + 1}: {e}. Splitting without stratification.")
             train_val_split = train_fold_dataset_full.train_test_split(
                 test_size=args.validation_split_percentage,
                 seed=args.seed + fold_idx
             )
             train_fold_dataset = train_val_split['train']
             val_fold_dataset = train_val_split['test']
             print(f"Inner split (non-stratified): {len(train_fold_dataset)} train, {len(val_fold_dataset)} validation samples.")

        test_fold_dataset = full_dataset.select(test_indices)
        
        training_args = TrainingArguments(
            output_dir=f'/projects/user/climate_glue_acl_Kfold/results/{args.task}/{args.model}_{args.seed}_{fold_idx}',
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_dir='/projects/user/climate_glue_acl_Kfold/logs',
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
            train_dataset=train_fold_dataset,
            eval_dataset=val_fold_dataset,
            class_weights=data_class.class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        trainer.save_model()
        predictions, label_ids, metrics = trainer.predict(val_fold_dataset)
        print(f'Metrics for val set: {metrics}')
        predictions, label_ids, metrics = trainer.predict(test_fold_dataset)
        result = {args.task: {'predictions': predictions, 'label_ids': label_ids, 'metrics': metrics}}

        with open(f'test_results/result_{args.task}_{args.model.replace("/","_")}_{args.seed}_{fold_idx}.pickle', 'wb') as f:
            pickle.dump(result, f)

        print(metrics)
        print(data_class.labels)
        wandb.log(metrics)

training_args = TrainingArguments(
    output_dir=f'/projects/user/climate_glue_acl_Kfold/results/{args.task}/{args.model}_{args.seed}',
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='/projects/user/climate_glue_acl_Kfold/logs',
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

with open(f'test_results/result_{args.task}_{args.model.replace("/","_")}_{args.seed}.pickle', 'wb') as f:
    pickle.dump(result, f)

print(metrics)
print(data_class.labels)
wandb.log(metrics)

