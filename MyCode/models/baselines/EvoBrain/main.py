import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import utils
from args import get_args
from collections import OrderedDict
from json import dumps
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import pandas as pd
import sklearn
import time
import torch
torch.autograd.set_detect_anomaly(True)

def main(args):

    
    args.cuda = torch.cuda.is_available()
    device = args.device if args.cuda else "cpu"

    
    utils.seed_torch(seed=args.rand_seed)

    
    args.save_dir = utils.get_save_dir(
        args.save_dir, args.dataset, args.task, args.max_seq_len, args.model_name, args.graph_type, args.rand_seed)
    
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    
    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))

    if args.model_name == "BIOT":
        args.use_fft = False

    
    log.info('Building dataset...')
    if args.dataset == 'CHBMIT':
        from data.dataloader_chb import load_dataset_chb
        print("Loading CHBMIT dataset...")
        dataloaders, datasets, scaler = load_dataset_chb(
            task = args.task,
            input_dir=args.input_dir,
            raw_data_dir=args.raw_data_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size,
            max_seq_len=args.max_seq_len,
            standardize=False,
            num_workers=args.num_workers,
            augmentation=args.data_augment,
            adj_mat_dir='.data/electrode_graph/adj_mx_3d.pkl',
            graph_type=args.graph_type,
            top_k=args.top_k,
            filter_type=args.filter_type,
            use_fft=args.use_fft,
            sampling_ratio=1,
            seed=123,
            preproc_dir=args.preproc_dir)
    else: 
        print("Loading TUSZ dataset...")
        if args.task == 'detection':
            from data.dataloader_detection import load_dataset_detection
            dataloaders, datasets, scaler = load_dataset_detection(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                time_step_size=args.time_step_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                augmentation=args.data_augment,
                adj_mat_dir='.data/electrode_graph/adj_mx_3d.pkl',
                graph_type=args.graph_type,
                top_k=args.top_k,
                filter_type=args.filter_type,
                use_fft=args.use_fft,
                sampling_ratio=1,
                seed=123,
                preproc_dir=args.preproc_dir)

        
        elif args.task == 'prediction':
            from data.dataloader_prediction import load_dataset_prediction
            dataloaders, datasets, scaler = load_dataset_prediction(
                input_dir=args.input_dir,
                raw_data_dir=args.raw_data_dir,
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                time_step_size=args.time_step_size,
                max_seq_len=args.max_seq_len,
                standardize=True,
                num_workers=args.num_workers,
                augmentation=args.data_augment,
                adj_mat_dir='.data/electrode_graph/adj_mx_3d.pkl',
                graph_type=args.graph_type,
                top_k=args.top_k,
                filter_type=args.filter_type,
                use_fft=args.use_fft,
                sampling_ratio=1,
                seed=123,
                preproc_dir=args.preproc_dir)
        else:
            raise NotImplementedError

    
    log.info('Building model...')
    if args.model_name == "dcrnn":
        from model.DCRNN import DCRNNModel_classification
        model = DCRNNModel_classification(
            args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "evolvegcn":
        from model.EGCN import EvolveGCN_Model_classification
        model = EvolveGCN_Model_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "evobrain":
        from model.EvoBrain import EvoBrain_classification
        if args.agg != "max":
            log.info("Using EvoBrain with aggregation method: {}".format(args.agg))
        model = EvoBrain_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "graphs4mer":
        from model.graphs4mer import GraphS4mer
        model = GraphS4mer(num_classes=args.num_classes, max_seq_len=args.max_seq_len, num_nodes=args.num_nodes)
    elif args.model_name == "gru_gcn":
        from model.gru_gcn import GRU_GCN_classification
        model = GRU_GCN_classification(args=args, num_classes=args.num_classes, device=device)
    elif args.model_name == "BIOT":
        from model.BIOT import BIOTClassifier
        args.use_fft = False
        model = BIOTClassifier(n_classes=args.num_classes, n_channels=args.num_nodes, n_fft=args.input_dim, hop_length=int(args.input_dim / 2))
    elif args.model_name == "lstm":
        from model.lstm import LSTMModel
        model = LSTMModel(args, args.num_classes, device)
    elif args.model_name == "cnnlstm":
        from model.cnnlstm import CNN_LSTM
        model = CNN_LSTM(args.num_classes, args.dataset)
    else:
        raise NotImplementedError

    if not args.test:
        if not args.fine_tune:
            if args.load_model_path is not None:
                model = utils.load_model_checkpoint(
                    args.load_model_path, model)
        else:  
            if args.load_model_path is not None:
                args_pretrained = copy.deepcopy(args)
                setattr(
                    args_pretrained,
                    'num_rnn_layers',
                    args.pretrained_num_rnn_layers)
                from model.DCRNN import DCRNNModel_nextTimePred
                pretrained_model = DCRNNModel_nextTimePred(
                    args=args_pretrained, device=device)  
                pretrained_model = utils.load_model_checkpoint(
                    args.load_model_path, pretrained_model)

                model = utils.build_finetune_model(
                    model_new=model,
                    model_pretrained=pretrained_model,
                    num_rnn_layers=args.num_rnn_layers)
            else:
                raise ValueError(
                    'For fine-tuning, provide pretrained model in load_model_path!')

        num_params = utils.count_parameters(model)
        log.info('Total number of trainable parameters: {}'.format(num_params))

        model = model.to(device)

        
        train(model, dataloaders, args, device, args.save_dir, log, tbx)

        
        best_path = os.path.join(args.save_dir, 'best.pth.tar')
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)

    else:
        if args.load_model_path is not None:
            model = utils.load_model_checkpoint(
                args.load_model_path, model)

    
    log.info('Training DONE. Evaluating model...')
    model = model.to(device)
    dev_results = evaluate(model,
                           dataloaders['dev'],
                           args,
                           args.save_dir,
                           device,
                           log,
                           is_test=True,
                           nll_meter=None,
                           eval_set='dev')

    dev_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                for k, v in dev_results.items())
    log.info('DEV set prediction results: {}'.format(dev_results_str))

    test_results = evaluate(model,
                            dataloaders['test'],
                            args,
                            args.save_dir,
                            device,
                            log,
                            is_test=True,
                            nll_meter=None,
                            eval_set='test',
                            best_thresh=dev_results['best_thresh'])

    
    test_results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                 for k, v in test_results.items())
    log.info('TEST set prediction results: {}'.format(test_results_str))


def train(model, dataloaders, args, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    """

    
    if (args.task == 'detection') or (args.task == 'prediction'):
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']

    
    saver = utils.CheckpointSaver(save_dir,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    
    model.train()

    
    budget_schedule = bool(getattr(args, 'budget_schedule', False))
    budget_discriminative_full_model = bool(
        getattr(args, 'budget_discriminative_full_model', False)
    )
    if budget_schedule:
        head_prefixes = tuple(getattr(args, 'budget_head_prefixes'))
        last_block_prefixes = tuple(getattr(args, 'budget_last_block_prefixes'))
        head_parameters = []
        last_block_parameters = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad = False
            if name.startswith(head_prefixes):
                parameter.requires_grad = True
                head_parameters.append(parameter)
            elif name.startswith(last_block_prefixes):
                last_block_parameters.append(parameter)
        if not head_parameters or not last_block_parameters:
            raise RuntimeError(
                'Budget fine-tuning could not resolve the classifier head or last feature block'
            )
        optimizer = optim.Adam(
            [
                {'params': head_parameters, 'lr': args.lr_init},
                {'params': last_block_parameters, 'lr': args.budget_backbone_lr},
            ],
            weight_decay=args.l2_wd,
        )
        log.info(
            'Budget fine-tuning schedule: head-only epochs=%d, total epochs=%d, '
            'head lr=%g, last-block lr=%g, patience=%d',
            args.budget_head_only_epochs,
            args.num_epochs,
            args.lr_init,
            args.budget_backbone_lr,
            args.patience,
        )
    elif budget_discriminative_full_model:
        head_parameters = []
        backbone_parameters = []
        for name, parameter in model.named_parameters():
            if name.startswith('fc.') or '.fc.' in name:
                parameter.requires_grad = True
                head_parameters.append(parameter)
            else:
                parameter.requires_grad = args.budget_head_only_epochs == 0
                backbone_parameters.append(parameter)
        if not head_parameters or not backbone_parameters:
            raise RuntimeError(
                'Budget full-model fine-tuning could not separate EvoBrain head and backbone'
            )
        optimizer = optim.Adam(
            [
                {'params': backbone_parameters, 'lr': args.budget_backbone_lr},
                {'params': head_parameters, 'lr': args.lr_init},
            ],
            weight_decay=args.l2_wd,
        )
        log.info(
            'Budget LP-FT: head-only epochs=%d, head lr=%g, '
            'backbone lr=%g, patience=%d',
            args.budget_head_only_epochs,
            args.lr_init,
            args.budget_backbone_lr,
            args.patience,
        )
    else:
        optimizer = optim.Adam(params=model.parameters(),
                               lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    
    nll_meter = utils.AverageMeter()

    
    log.info('Training...')
    epoch = 0
    step = 0
    best_early_stop_metric = -float('inf') if args.maximize_metric else float('inf')
    min_delta = float(getattr(args, 'min_delta', 0.0))
    patience_count = 0
    early_stop = False
    memory_usage_list = []
    time_list = []
    while (epoch != args.num_epochs) and (not early_stop):
        epoch += 1
        if budget_schedule and epoch == args.budget_head_only_epochs + 1:
            for name, parameter in model.named_parameters():
                if name.startswith(last_block_prefixes):
                    parameter.requires_grad = True
            log.info('Budget fine-tuning stage 2 started: classifier head and last feature block')
        if (
            budget_discriminative_full_model
            and args.budget_head_only_epochs > 0
            and epoch == args.budget_head_only_epochs + 1
        ):
            for parameter in backbone_parameters:
                parameter.requires_grad = True
            log.info('Budget LP-FT full-model phase started')
        log.info('Starting epoch {}...'.format(epoch))
        total_batches = len(train_loader)
        sampler = getattr(train_loader, 'sampler', None)
        sampled_clips = len(sampler) if sampler is not None else len(train_loader.dataset)
        log.info(
            'Epoch %d effective training clips=%d batches=%d batch_size=%d',
            epoch,
            sampled_clips,
            total_batches,
            train_loader.batch_size,
        )
        with torch.enable_grad(), \
                tqdm(total=total_batches, unit='batch', colour='green') as progress_bar:
            for x, y, seq_lengths, supports, adj, file_name in train_loader:
                batch_size = x.shape[0]

                
                x = x.to(device)
                y = y.view(-1).to(device)  
                seq_lengths = seq_lengths.view(-1).to(device)  
                supports = supports.to(device)
                adj = adj.to(device)

                
                optimizer.zero_grad()

                
                
                start_time = time.time()
                initial_memory = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0

                if args.model_name == "evobrain" or args.model_name == "evolvegcn" or args.model_name == "gru_gcn":
                    logits, _ = model(x, seq_lengths, adj)
                elif args.model_name == "dcrnn":
                    logits, _ = model(x, seq_lengths, supports)     
                elif args.model_name == "BIOT":
                    logits, _ = model(x)  
                elif args.model_name == "lstm" or args.model_name == "cnnlstm" or args.model_name == "graphs4mer":
                    logits, _ = model(x, seq_lengths)
                else:
                    print("model_name: ", args.model_name)
                    raise NotImplementedError
                if logits.shape[-1] == 1:
                    logits = logits.view(-1)          
                loss = loss_fn(logits, y)
                loss_val = loss.item()

                
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()

                end_time = time.time()
                max_memory = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0

                memory_usage_list.append(max_memory - initial_memory)
                time_list.append(end_time - start_time)

                step += batch_size

                
                progress_bar.update(1)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         lr=optimizer.param_groups[0]['lr'])

                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)
                if (args.stop == True) and (len(time_list) > 1000):
                    break

            if epoch % args.eval_every == 0:
                
                log.info('Evaluating at epoch {}...'.format(epoch))
                nll_meter.reset()
                eval_results = evaluate(model,
                                        dev_loader,
                                        args,
                                        save_dir,
                                        device,
                                        log,
                                        is_test=False,
                                        nll_meter=nll_meter)
                best_path = saver.save(epoch,
                                       model,
                                       optimizer,
                                       eval_results[args.metric_name])

                
                current_metric = eval_results[args.metric_name]
                improved = (
                    current_metric > best_early_stop_metric + min_delta
                    if args.maximize_metric
                    else current_metric < best_early_stop_metric - min_delta
                )
                early_stopping_active = (
                    (not budget_schedule and not budget_discriminative_full_model)
                    or epoch > args.budget_head_only_epochs
                )
                if improved:
                    patience_count = 0
                    best_early_stop_metric = current_metric
                elif early_stopping_active:
                    patience_count += 1

                
                if early_stopping_active and patience_count >= args.patience:
                    early_stop = True

                
                model.train()

                
                results_str = ', '.join('{}: {:.3f}'.format(k, v)
                                        for k, v in eval_results.items())
                log.info('Dev {}'.format(results_str))

                
                log.info('Visualizing in TensorBoard...')
                for k, v in eval_results.items():
                    tbx.add_scalar('eval/{}'.format(k), v, step)

        
        scheduler.step()

    max_memory_usage = np.max(memory_usage_list) / (1024 ** 2)  
    avg_time_per_batch = np.mean(time_list)

    log.info(f"Average Training Time per Batch: {avg_time_per_batch:.4f} seconds")


def evaluate(
        model,
        dataloader,
        args,
        save_dir,
        device,
        log,
        is_test=False,
        nll_meter=None,
        eval_set='dev',
        best_thresh=0.5):
    
    model.eval()

    
    if (args.task == 'detection') or (args.task == 'prediction'):
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    hidden_all = []
    time_list = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset), colour="green") as progress_bar:
        for x, y, seq_lengths, supports, adj, file_name in dataloader:
            batch_size = x.shape[0]

            
            x = x.to(device)
            y = y.view(-1).to(device)  
            seq_lengths = seq_lengths.view(-1).to(device)  
            supports = supports.to(device)
            adj = adj.to(device)

            start_time = time.time()
            
            
            if args.model_name == "evobrain":
                logits, hidden = model(x, seq_lengths, adj)
            elif args.model_name == "gru_gcn":
                logits, hidden = model(x, seq_lengths, adj)
            elif args.model_name == "dcrnn":
                logits, hidden = model(x, seq_lengths, supports)
            elif args.model_name == "evolvegcn":
                logits, hidden = model(x, seq_lengths, adj)
            elif args.model_name == "BIOT":
                logits, hidden = model(x)
            elif args.model_name == "lstm" or args.model_name == "cnnlstm" or args.model_name == "graphs4mer":
                logits, hidden = model(x, seq_lengths)
            else:
                raise NotImplementedError

            if args.num_classes == 1:  
                logits = logits.view(-1)  
                y_prob = torch.sigmoid(logits).cpu().numpy()  
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > best_thresh).astype(int)  
            else:
                
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)  
                y_true = y.cpu().numpy().astype(int)
            
            time_list.append(time.time() - start_time)
            

            
            loss = loss_fn(logits, y)
            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            file_name_all.extend(file_name)
            hidden_all.append(hidden.cpu().reshape(hidden.shape[0], -1))

            
            progress_bar.update(batch_size)
            if (args.stop == True) and (len(time_list) > 1000):
                break

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    hidden_all = np.concatenate(hidden_all, axis=0)
    
    

    
    if is_test:
        results_file = os.path.join(save_dir, f'{eval_set}_results.npz')
        np.savez(results_file, 
                 y_true=y_true_all, 
                 y_pred=y_pred_all, 
                 y_prob=y_prob_all, 
                 file_names=file_name_all)
        print(f"Evaluation results saved to {results_file}")

    if eval_set=='test':
            output_file = os.path.join(save_dir, "hidden.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            
            df = pd.DataFrame(hidden_all)
            
            df.to_csv(output_file, mode='w', header=False, index=False)

            output_file = os.path.join(save_dir, "true_labels.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            
            df = pd.DataFrame(np.expand_dims(y_true_all, axis=0))
            
            df.to_csv(output_file, mode='w', header=False, index=False)
    
    avg_time_per_batch = np.mean(time_list)
    log.info(f"Average Test Time per Batch: {avg_time_per_batch:.4f} seconds")


    
    if ((args.task == "detection") or (args.task == "prediction")) and (eval_set == 'dev') and is_test:
        best_thresh = utils.thresh_max_f1(y_true=y_true_all, y_prob=y_prob_all)
        
        y_pred_all = (y_prob_all > best_thresh).astype(int)  
    else:
        best_thresh = best_thresh

    scores_dict, _, _ = utils.eval_dict(y_pred=y_pred_all,
                                        y=y_true_all,
                                        y_prob=y_prob_all,
                                        file_names=file_name_all,
                                        average="binary" if ((args.task == "detection")or(args.task == "prediction")) else "weighted")

    if args.num_classes == 1 and is_test:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true_all, y_prob_all)
        roc_file = os.path.join(save_dir, f'{eval_set}_roc_data.npz')
        np.savez(roc_file, fpr=fpr, tpr=tpr, thresholds=thresholds)
        print(f"ROC curve data saved to {roc_file}")

    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()
    results_list = [('loss', eval_loss),
                    ('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision']),
                    ('best_thresh', best_thresh)]
    if 'auroc' in scores_dict.keys():
        results_list.append(('auroc', scores_dict['auroc']))
    results = OrderedDict(results_list)

    return results

def check_tensor(data, description):
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"{description} is not a tensor! Found type: {type(data)}")


if __name__ == '__main__':
    main(get_args())
