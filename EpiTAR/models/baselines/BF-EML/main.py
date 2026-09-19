import os
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path
from data import MultiViewDataset
from models import EML
import pickle

BASE_DIR = Path(__file__).resolve().parent
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def pretrain_model(device, train_dataset, valid_dataset, epochs, saving_path, num_classes):
    
    data_train = train_dataset
    data_valid = valid_dataset
    train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=1024, shuffle=False)

    
    model = EML(sample_shapes=[s.shape for s in data_train[0]['x'].values()], num_classes=num_classes, device=device)
    model.ModelLoss = 'EML'

    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.1)

    best_valid_acc = 0.
    best_model_wts = model.state_dict()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, num_samples = 0, 0, 0
        for batch in train_loader:
            x, target = batch['x'], batch['y']
            for v in x.keys():
                x[v] = x[v].to(device)
            target = target.to(device)
            view_e, fusion_e, loss, view_h = model(x, target, kl_penalty=min(1., epoch / 20))
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            train_loss += loss.mean().item() * len(target)
            correct += torch.sum(fusion_e.argmax(dim=-1).eq(target)).item()
            num_samples += len(target)
        scheduler.step()
        train_loss = train_loss / num_samples
        train_acc = correct / num_samples
        val = validate(device, model, valid_loader)
        if best_valid_acc < val['accuracy']:
            best_valid_acc = val['accuracy']
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch:2d}; train loss {train_loss:.4f}, train acc {train_acc:.4f};', end=' ')
        if num_classes == 2:
            print('validation:', *[f'{v:.4f}' for k, v in val.items() if k.startswith('b_')])
        else:
            print('validation:', val['accuracy'])

    model.load_state_dict(best_model_wts)
    val = validate(device, model, valid_loader)
    if saving_path is not None:
        os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        torch.save(model.state_dict(), saving_path)
    model.best_valid_acc = best_valid_acc
    print('Validation for best model:', *[f'{k}:{v:.6f}' for k, v in val.items()])
    return model


def split_data_by_evidence(dataset, model, batch_size=64, num_workers=8, device='cuda', high_ratio=0.8):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=False)

    all_evidences = []
    all_indices = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            inputs = batch['x']
            targets = batch['y']
            indices = batch['index']

            inputs = {key: val.to(device) for key, val in inputs.items()}
            _, fusion_e, _, _ = model(inputs)

            evidence_scores = fusion_e.max(dim=1)[0]
            batch_indices = indices

            all_evidences.append(evidence_scores)
            all_indices.append(batch_indices)
            all_labels.append(targets)


    all_evidences = torch.cat(all_evidences).to(device)
    all_indices = torch.cat(all_indices).to(device)
    all_labels = torch.cat(all_labels).to(device)


    positive_indices = all_indices[all_labels == 1]
    negative_indices = all_indices[all_labels == 0]

    positive_evidences = all_evidences[all_labels == 1]
    negative_evidences = all_evidences[all_labels == 0]

    sorted_positive_evidences, sorted_positive_indices = torch.sort(positive_evidences, descending=True)
    sorted_negative_evidences, sorted_negative_indices = torch.sort(negative_evidences, descending=True)

    high_positive_cutoff = int(high_ratio * len(sorted_positive_indices))
    high_negative_cutoff = int(high_ratio * len(sorted_negative_indices))

    high_evidence_positive_indices = positive_indices[sorted_positive_indices[:high_positive_cutoff]].tolist()
    low_evidence_positive_indices = positive_indices[sorted_positive_indices[high_positive_cutoff:]].tolist()


    low_positive_sample_size = len(low_evidence_positive_indices)
    low_negative_sample_size = int(low_positive_sample_size * 0.8)

    high_evidence_negative_indices = negative_indices[sorted_negative_indices[:high_negative_cutoff]].tolist()
    low_evidence_negative_indices = negative_indices[sorted_negative_indices[high_negative_cutoff:high_negative_cutoff + low_negative_sample_size]].tolist()

    high_evidence_indices = high_evidence_positive_indices + high_evidence_negative_indices
    low_evidence_indices = low_evidence_positive_indices + low_evidence_negative_indices


    high_evidence_dataset = Subset(dataset, high_evidence_indices)
    low_evidence_dataset = Subset(dataset, low_evidence_indices)

    return high_evidence_dataset, low_evidence_dataset

def check_dataset_balance(dataset):
    labels = []
    for data in dataset:
        labels.append(data['y'].item())
    labels = torch.tensor(labels)
    unique, counts = torch.unique(labels, return_counts=True)
    print("Labels distribution in dataset:", dict(zip(unique.tolist(), counts.tolist())))


def train_final_model(device, train_dataset, valid_dataset, ModelBias, epochs, saving_path):
    
    
    
    data_train = train_dataset
    data_valid = valid_dataset
    num_classes = len(set(data_train.y))
    train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=1024, shuffle=False)

    
    model = EML(sample_shapes=[s.shape for s in data_train[0]['x'].values()], num_classes=num_classes, device=device)
    model.ModelLoss = 'HSIC'
    model.ModelPre = ModelBias

    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.1)

    print("Training final model...")

    best_valid_acc = 0.
    best_model_wts = model.state_dict()
    for epoch in range(epochs):
        model.train()
        train_loss, correct, num_samples = 0, 0, 0
        for batch in train_loader:
            x, target = batch['x'], batch['y']
            for v in x.keys():
                x[v] = x[v].to(device)
            target = target.to(device)
            _, fusion_e, loss, _ = model(x, target, kl_penalty=min(1., epoch / 20), ModelLoss='HSIC', iter_num=epoch)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            train_loss += loss.mean().item() * len(target)
            correct += torch.sum(fusion_e.argmax(dim=-1).eq(target)).item()
            num_samples += len(target)
        scheduler.step()
        train_loss = train_loss / num_samples
        train_acc = correct / num_samples
        val = validate(device, model, valid_loader)
        if best_valid_acc < val['accuracy']:
            best_valid_acc = val['accuracy']
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch:2d}; train loss {train_loss:.4f}, train acc {train_acc:.4f};', end=' ')
        if num_classes == 2:
            print('validation:', *[f'{v:.4f}' for k, v in val.items() if k.startswith('b_')])
        else:
            print('validation:', val['accuracy'])

    model.load_state_dict(best_model_wts)
    val = validate(device, model, valid_loader)
    if saving_path is not None:
        os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        torch.save(model.state_dict(), saving_path)
    model.best_valid_acc = best_valid_acc  
    print('Validation for best model:', *[f'{k}:{v:.6f}' for k, v in val.items()])
    return model




def validate(device, model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = num_samples = 0  
        TP = TN = FP = FN = 0  
        for batch in dataloader:
            x, y = batch['x'], batch['y']
            for v in x.keys():
                x[v] = x[v].to(device)
            view_e, fusion_e, loss, view_h = model(x)
            pred = fusion_e.cpu().argmax(dim=-1)
            correct += torch.sum(pred == y).item()
            num_samples += len(y)
            TP += torch.sum((pred == 1) & (pred == y)).item()
            TN += torch.sum((pred == 0) & (pred == y)).item()
            FP += torch.sum((pred == 1) & (pred != y)).item()
            FN += torch.sum((pred == 0) & (pred != y)).item()
    accuracy = correct / num_samples
    b_accuracy = (TN + TP) / (TP + TN + FP + FN)
    b_sensitivity = TP / (TP + FN)
    b_specificity = TN / (TN + FP)
    return {
        'accuracy': accuracy,
        'b_accuracy': b_accuracy, 'b_sensitivity': b_sensitivity, 'b_specificity': b_specificity
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_path = BASE_DIR / 'dataset/eeg/data/domain_feature/data23_fold0_train.pkl'
    valid_path = BASE_DIR / 'dataset/eeg/data/domain_feature/data23_fold0_valid.pkl'
    epochs = 20
    saving_path = BASE_DIR / 'saved_models'
    pretrain_mode = 'none'
    modelbias_mode = 'none'

    
    print("Loading training dataset...")
    data_train = MultiViewDataset(data_path=train_path)
    data_valid = MultiViewDataset(data_path=valid_path)
    num_classes = len(set(data_train.y))
    sample_shapes = [s.shape for s in data_train[0]['x'].values()]
    print("Training dataset loaded.")


    if pretrain_mode == 'none':
        
        print("Starting pretraining of ModelPre...")
        ModelPre = pretrain_model(device, data_train, data_valid, epochs, saving_path / '0809_080_ModelPreMerged_data_Only23.pth', num_classes)
        print("Finished pretraining of ModelPre.")

    elif pretrain_mode == 'ready':
        
        print("Initializing ModelPre...")
        ModelPre = EML(sample_shapes=sample_shapes, num_classes=num_classes, device=device)
        print("ModelPre initialized.")

        print("Loading pre-trained ModelPre...")
        model_pre_path = saving_path / '0803_075_ModelPreMerged_data_exclude1.pth'
        ModelPre.load_state_dict(torch.load(model_pre_path, map_location=device))
        ModelPre.to(device)
        ModelPre.eval()
        print("Pre-trained ModelPre loaded.")

    if modelbias_mode == 'none':
        
        print("Starting data splitting based on evidence...")
        high_evidence_dataset, low_evidence_dataset = split_data_by_evidence(data_train, ModelPre, device=device, high_ratio=0.9)
        print("Finished data splitting.")


        print("Checking low evidence dataset...")
        check_dataset_balance(low_evidence_dataset)

        print("Checking high evidence dataset...")
        check_dataset_balance(high_evidence_dataset)


        
        print("Starting training of ModelBias with low evidence data...")
        ModelBias = pretrain_model(device, low_evidence_dataset, data_valid, epochs, saving_path / '0809_080_ModelBiasMerged_data_Only23.pth', num_classes)
        print("Finished training of ModelBias.")


    elif modelbias_mode == 'ready':
        print("Initializing ModelBias...")
        ModelBias = EML(sample_shapes=sample_shapes, num_classes=num_classes, device=device)
        print("ModelBias initialized.")

        print("Loading biased-trained ModelBias...")
        model_bias_path = saving_path / 'ModelBiasTry.pth'
        ModelBias.load_state_dict(torch.load(model_bias_path, map_location=device))
        ModelBias.to(device)
        ModelBias.eval()
        print("Bias-trained ModelBias loaded.")

    
    print("Starting training of final model using HSIC...")
    model = EML(sample_shapes=sample_shapes, num_classes=num_classes, device=device)
    model.ModelLoss = 'HSIC'
    model.ModelPre = ModelBias
    epochs = 40



    print("Starting training of final model using HSIC...")
    train_final_model(device, data_train, data_valid, ModelBias, epochs, saving_path / '0800_080_ModelFinalMerged_data_2003.pth')
    print("Finished training of final model.")


if __name__ == '__main__':
    main()



