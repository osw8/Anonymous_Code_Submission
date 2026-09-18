import argparse
import yaml
import os
import numpy as np
import pandas as pd
import time

from sklearn.metrics import (
    cohen_kappa_score,
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score
)
from models.EEG_decoding_pipelines import (Scaler3D, count_pipeline_params)

from utils import get_downstream_task_info, split_recordings_for_evaluation, get_dataset, stack_by_class_numpy, get_ML_models
from torch.utils.data import ConcatDataset

HEADERS = [ 'evaluation_scheme',
    'model','fold', 'optimizer_spec', 'current_subject', 'test_subject',
    'kappa',
    'auc', 
    'balanced_acc',
    'acc1',
    'acc2',
    'n_parameters', 'train_runtime'
]


def append_to_csv(csv_path: str, headers: list, row: dict):
    
    if not os.path.isfile(csv_path):
        print(f"File '{csv_path}' not found. Creating new file with headers.")
        pd.DataFrame(columns=headers).to_csv(csv_path, index=False)

    
    df = pd.read_csv(csv_path)

    
    missing = set(headers) - set(row.keys())
    if missing:
        raise ValueError(f"New entry is missing columns: {missing}")

    
    new_df = pd.DataFrame([row], columns=headers)
    df = pd.concat([df, new_df], ignore_index=True)

    
    df.to_csv(csv_path, index=False)
    print(f"Added new entry for subject '{row.get('test_subject')}'. File saved.")

def top_k_accuracy(y_true, y_pred_proba, k=1):
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    matches = np.any(top_k_preds == y_true[:, np.newaxis], axis=1)
    return np.mean(matches)

def get_args_parser():
    parser = argparse.ArgumentParser('EEG Classification pipeline', add_help=False)
    
    
    parser.add_argument('--seed', default = 0, type = int)
    
    
    parser.add_argument('--model', default='fbcsp', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_data_transform', default='None', type=str,
                        help='data transformation')

    
    parser.add_argument('--dataset_yaml', default="dataset_specs.yaml", type=str,
                        help='dataset yaml with dataset specs')
    parser.add_argument('--downstream_task_yaml', default="downstream_task_specs.yaml", type=str,
                        help='dataset yaml with dataset specs')
    parser.add_argument('--downstream_task', default="upper_limb_motorexecution", type=str,
                        help='which downstream task to benchmark')
    parser.add_argument('--evaluation_scheme', default="population", type=str,
                        help='which training-finetuning-testing scheme to use')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')

    return parser

def evaluate(args):
    
    args = get_downstream_task_info(args)
    
    
    experiment_run_split = split_recordings_for_evaluation(args)

    print(f"There are {experiment_run_split.get_number_of_runs()} of runs in this experiment!", flush=True)
    
    
    pipeline, CSV_PATH = get_ML_models(args)
    
    for run_idx in range(experiment_run_split.get_number_of_runs()):
        np.random.seed(args.seed)

        
        run_name = experiment_run_split.get_run_description(run_idx)
        current_run_sub_of_interested = run_name.split("sub-")[-1]
        loso_left = ['Subject3', 'Subject18', 'Subject8', 'Subject13','Subject2','Subject30','Subject10','Subject20', 'Subject29',
                        'Subject5','Subject1','Subject12','Subject32','Subject4']
        persub_left = ['Subject30','Subject10','Subject20', 'Subject29',
                        'Subject5','Subject1','Subject12','Subject32','Subject4']
                        
        if current_run_sub_of_interested in persub_left:
            this_run_split = experiment_run_split.get_run(run_idx)
            print(f"current run: {run_name}, train on {current_run_sub_of_interested}, total: {args.fold} folds", flush=True)
        
            
            for fold in range(args.fold):
                
                trainsets, finetunesets, testsets = get_dataset(args, fold, this_run_split)
                all_train_set = ConcatDataset(trainsets)
                if finetunesets:
                    all_finetune_set = ConcatDataset(finetunesets)
                all_test_set = ConcatDataset(testsets)
                
                
                print(f"fold: {fold}", flush=True)
    
                train_data = stack_by_class_numpy(all_train_set)
                test_data = stack_by_class_numpy(all_test_set)
                if finetunesets:
                    finetune_data = stack_by_class_numpy(all_finetune_set)
        
                print(f"There are {args.nb_classes} number of classes", flush=True)
                for cls_ctr in range(args.nb_classes):
                    print(f"class {cls_ctr}, training data: {train_data[cls_ctr].shape}, test data: {test_data[cls_ctr].shape}", flush=True)
                
                del trainsets, finetunesets
                
                
                
                all_data = []
                all_label = []
                for label, data in sorted(train_data.items()):
                    all_data.append(data)
                    num_samples = data.shape[0]
                    all_label.append(np.full(num_samples, label))
        
                X_train = np.concatenate(all_data, axis=0).astype(np.float64)
                y_train = np.concatenate(all_label, axis=0)
        
                
                norm = Scaler3D()
                norm.fit(X_train)
                X_train = norm.transform(X_train)
                
                
                start = time.time()
                pipeline.fit(X_train, y_train)
                end = time.time()
                
                
                
                
                
                all_data = []
                all_label = []
                for label, data in sorted(test_data.items()):
                    all_data.append(data)
                    num_samples = data.shape[0]
                    all_label.append(np.full(num_samples, label))
        
                X_test = np.concatenate(all_data, axis=0).astype(np.float64)
                y_test = np.concatenate(all_label, axis=0)
                 
                
                X_test = norm.transform(X_test)
                
                print(f'X_test shape: {X_test.shape}')
                
                
                y_pred_proba = pipeline.predict_proba(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
        
                
                KAPPA = cohen_kappa_score(y_test, y_pred)
        
                
                BACC = balanced_accuracy_score(y_test, y_pred)
        
                
                AUC = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') 
                
        
                
                TOP1ACC = accuracy_score(y_test, y_pred)
        
                
                TOP2ACC = top_k_accuracy(y_test, y_pred_proba, k=2)
        
                
                trainTime = end - start
        
                
                NoParameter = count_pipeline_params(pipeline)
        
                
                new_entry = {
                    'evaluation_scheme': args.evaluation_scheme,
                    'model': args.model,
                    'fold': fold,
                    'optimizer_spec': [],
                    'current_subject': current_run_sub_of_interested,
                    'test_subject': 'whole',
                    'kappa': KAPPA,
                    'auc': AUC,
                    'balanced_acc': BACC,
                    'acc1': TOP1ACC,
                    'acc2': TOP2ACC,
                    'n_parameters': NoParameter,
                    'train_runtime': trainTime
                }
                append_to_csv(CSV_PATH, HEADERS, new_entry)
        
                
                
                for single_test_set in testsets:
                    this_sub_test_data = stack_by_class_numpy(single_test_set)
                    
        
                    
                    
                    all_data = []
                    all_label = []
                    for label, data in sorted(this_sub_test_data.items()):
                        all_data.append(data[:, ssvep_channel_idx, :])
                        num_samples = data.shape[0]
                        all_label.append(np.full(num_samples, label))
        
                    X_test = np.concatenate(all_data, axis=0).astype(np.float64)
                    y_test = np.concatenate(all_label, axis=0)
        
                    
                    X_test = norm.transform(X_test)
        
                    
                    y_pred_proba = pipeline.predict_proba(X_test)
                    y_pred = np.argmax(y_pred_proba, axis=1)
        
                    
                    KAPPA = cohen_kappa_score(y_test, y_pred)
        
                    
                    BACC = balanced_accuracy_score(y_test, y_pred)
        
                    
                    AUC = roc_auc_score(y_test, y_pred_proba, multi_class='ovr') 
                    
        
                    
                    TOP1ACC = accuracy_score(y_test, y_pred)
        
                    
                    TOP2ACC = top_k_accuracy(y_test, y_pred_proba, k = 2)
        
                    
                    trainTime = end - start
        
                    
                    NoParameter = count_pipeline_params(pipeline)
    
                    
                    new_entry = {
                        'evaluation_scheme': args.evaluation_scheme,
                        'model': args.model,
                        'fold': fold,
                        'optimizer_spec': [],
                        'current_subject': current_run_sub_of_interested,
                        'test_subject': single_test_set.subjectName,
                        'kappa': KAPPA,
                        'auc': AUC,
                        'balanced_acc': BACC,
                        'acc1': TOP1ACC,
                        'acc2': TOP2ACC,
                        'n_parameters': NoParameter,
                        'train_runtime': trainTime
                    }
                    append_to_csv(CSV_PATH, HEADERS, new_entry)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    evaluate(args)
