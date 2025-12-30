# -*- coding: utf-8 -*-
"""
@author: P Xia
"""
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
import copy
import os
import random
import datasets
import models
from datasets.PreprocessedDataset import create_train_val_datasets, PreprocessedMultiConditionDataset
from utils import write_log, compute_accuracy
import torch.nn as nn

#working condition
WC = {0:"MS20_LS",
      1:"MS30_LS",
      2:"MS40_LS",
      3:"MS45_LS",
      4:"MS30_LV2",
      5:"MS30_LV20",
      6:"MV2.5_LS",
      7:"MV5_LS",
      8:"MV10_LS"}

def init_seed(opt):
#    torch.cuda.cudnn_enabled = False
    random.seed(opt.manual_seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, set_mode, data_dir, num_samples, val_ratio=0.2):
    # Dataset = getattr(datasets, opt.data_name)
    full_set = []
    if set_mode == 'src':
        # Create training and validation datasets for each source domain
        for cond in opt.source_condition:
            dst = {}
            dst['train'], dst['val'] = create_train_val_datasets(
                data_dir=data_dir,
                conditions=[WC[cond]],
                transform=None,  # optional data transformations
                samples_per_label=num_samples,  # loaded samples per label
                val_ratio=val_ratio,  # validation set ratio
                random_seed=42  # random seed for reproducibility
            )
            full_set.append(dst)
    elif set_mode == 'tar':
        condition = opt.target_condition
        dst = {}
        dst['test'] = PreprocessedMultiConditionDataset(
            data_dir=data_dir,
            conditions=[WC[i] for i in condition],
            transform=None, 
            samples_per_label=num_samples,
            random_seed=42
        )
        full_set.append(dst)
    else:
        raise Exception("src or tar flag is not implement")

    return full_set


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def gaussian_kernel(x, y, gamma=1.0):
    """Compute the Gaussian kernel between two sets of samples."""
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    xx = xx.diag().unsqueeze(0).expand_as(xx)
    yy = yy.diag().unsqueeze(0).expand_as(yy)
    
    kernel = torch.exp(-gamma * (xx.t() + yy - 2 * xy))
    return kernel


def mmd_loss(x, y):
    """Compute the Maximum Mean Discrepancy (MMD) loss between two sets of samples."""
    x_kernel = gaussian_kernel(x, x)
    y_kernel = gaussian_kernel(y, y)
    xy_kernel = gaussian_kernel(x, y)
    
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def orthogonality_constraint(x, y):
    """Compute the orthogonality constraint loss between two feature matrices.
    Args:
        x: 1st feature matrix [batch_size, feature_dim]
        y: 2nd feature matrix [batch_size, feature_dim]
    Returns:
        Constraint loss: mean((X_norm^T Y_norm)^2), X_norm and Y_norm are features after zero-mean and L2-normalization
    """
    batch_size = x.size(0)
    # Flatten the feature matrices
    x = x.view(batch_size, -1)
    y = y.view(batch_size, -1)
    
    # Zero-mean
    x_mean = torch.mean(x, dim=0, keepdim=True)
    y_mean = torch.mean(y, dim=0, keepdim=True)
    x = x - x_mean
    y = y - y_mean
    
    # L2-normalization
    x_l2_norm = torch.norm(x, p=2, dim=1, keepdim=True).detach()
    x_l2 = x.div(x_l2_norm.expand_as(x) + 1e-6)
    
    y_l2_norm = torch.norm(y, p=2, dim=1, keepdim=True).detach()
    y_l2 = y.div(y_l2_norm.expand_as(y) + 1e-6)
    
    # Compute the orthogonality constraint loss
    diff_loss = torch.mean((x_l2.t().mm(y_l2)).pow(2))
    
    return diff_loss

def covariance_loss(x, y):
    """Compute the covariance loss between two feature matrices."""
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov = torch.mm(x.t(), y) / (x.size(0) - 1)
    return torch.norm(cov, p=2)

def val_workflow(src_val_dataloader, opt, model, classifier, domain_extractor, ite, best_accuracy_val):
    model.eval()
    classifier.eval()
    domain_extractor.eval()
    accuracies = []
    for index in range(len(src_val_dataloader)):
        val_preds = []
        val_labels = []
        for batch_idx, (inputs, labels) in enumerate(src_val_dataloader[index]):
            vibs_val, curs_val, auds_val = inputs['vibration'], inputs['current'], inputs['audio']
            
            vibs_val, curs_val, auds_val, labels = vibs_val.cuda(), curs_val.cuda(), auds_val.cuda(), labels.cuda()
            
            with torch.no_grad():
                feas_vib, feas_cur, feas_aud = model(vibs_val, curs_val, auds_val)
                domain_feat = domain_extractor(feas_vib, feas_cur, feas_aud)
                outputs_val = classifier(domain_feat)
                
                predictions = outputs_val.cpu().data.numpy()
                val_preds.append(predictions)
                val_labels.append(labels.cpu().numpy())
            
        predictions = np.concatenate(val_preds)
        labels = np.concatenate(val_labels)

        accuracy_val = compute_accuracy(predictions=predictions, labels=labels)
        accuracies.append(accuracy_val)

    mean_acc = np.mean(accuracies)

    if mean_acc > best_accuracy_val:
        best_accuracy_val = mean_acc

        f = open(os.path.join(opt.experiment_root, 'Best_val.txt'), mode='a')
        f.write('ite:{}, best val accuracy:{}\n'.format(ite, best_accuracy_val))
        f.close()

        if not os.path.exists(opt.experiment_root):
            os.mkdir(opt.experiment_root)

        # outfile = os.path.join(opt.experiment_root, 'best_model.pth')
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'classifier_state_dict': classifier.state_dict(),
        #     'domain_extractor_state_dict': domain_extractor.state_dict(),
        #     'ite': ite
        # }, outfile)
    print('ite:{}, val accuracy:{}\n'.format(ite, mean_acc))
    
    return best_accuracy_val


def train_mix(opt, src_dataset, model, classifier, domain_extractor, optim, lr_scheduler):
    model.train()
    classifier.train()
    domain_extractor.train()
    
    src_tr_dataloader = []
    for dst in src_dataset:
        tr_dataloader = torch.utils.data.DataLoader(dst['train'],
                                                    batch_size = opt.batch_size,
                                                    # num_workers=4,
                                                    # pin_memory=True,
                                                    shuffle=True)
        src_tr_dataloader.append(tr_dataloader)
    
    src_val_dataloader = []
    for dst in src_dataset:
        val_dataloader = torch.utils.data.DataLoader(dst['val'],
                                                     batch_size = opt.batch_size,
                                                    #  num_workers=4,
                                                    #  pin_memory=True,
                                                     shuffle=False)
        src_val_dataloader.append(val_dataloader)

    best_accuracy_val = -1
    loss_fn = torch.nn.CrossEntropyLoss()

    for ite in range(opt.epochs):
        train_loss = 0.0
        train_preds = []
        train_labels = []
        model.train()
        classifier.train()
        domain_extractor.train()
        
        pbar = tqdm(range(min(len(dataloader) for dataloader in src_tr_dataloader)), 
                   desc=f'Epoch {ite+1}/{opt.epochs}')
        # Initialize iterators for each dataloader
        iterators = [iter(dl) for dl in src_tr_dataloader]
        
        for batch_idx in pbar:
            # Store data and features from all domains
            all_domain_data = []
            all_domain_labels = []
            all_domain_features = []
            domain_features = []
            batch_preds = []
            batch_labels = []
            cls_loss = 0.0
            
            # Get one batch from each domain
            for domain_idx, it in enumerate(iterators):
                try:
                    images_train, labels_train = next(it)
                
                    vibs_train, curs_train, auds_train = images_train['vibration'], \
                                                        images_train['current'], \
                                                        images_train['audio']

                    vibs_train, curs_train, auds_train, labels_train = vibs_train.cuda(), \
                                                                    curs_train.cuda(), \
                                                                    auds_train.cuda(), \
                                                                    labels_train.cuda().long()

                    # feas_vib, feas_cur, feas_aud = model(vibs_train, curs_train, auds_train)
                    
                    # store the data and labels for this domain
                    all_domain_data.append((vibs_train, curs_train, auds_train))
                    all_domain_labels.append(labels_train)
                    # all_domain_features.append((feas_vib, feas_cur, feas_aud))

                except StopIteration:
                    iterators[domain_idx] = iter(src_tr_dataloader[domain_idx])
                    continue
            
            alpha = 0.5  # Parameter for Beta distribution
            num_domains = len(all_domain_data)
            # Process each domain
            for domain_idx in range(num_domains):
                labels_train = all_domain_labels[domain_idx]
                # feas_vib, feas_cur, feas_aud = all_domain_features[domain_idx]
                datas_vib, datas_cur, datas_aud = all_domain_data[domain_idx]

                if num_domains > 1:
                    batch_size = datas_vib.size(0)
                    # batch_size = feas_vib.size(0)
                    # clone features
                    # new_feas_vib = feas_vib.clone()
                    # new_feas_cur = feas_cur.clone()
                    # new_feas_aud = feas_aud.clone()
                    new_datas_vib = datas_vib.clone()
                    new_datas_cur = datas_cur.clone()
                    new_datas_aud = datas_aud.clone()
                   
                    # Preprocessing: build a dictionary mapping from label to list of indices for each other domain
                    label_to_indices = {}
                    for d in range(num_domains):
                        if d != domain_idx:
                            d_labels = all_domain_labels[d]
                            label_to_indices[d] = {}
                            for j, label in enumerate(d_labels):
                                if j < batch_size:
                                    label_val = label.item()
                                    if label_val not in label_to_indices[d]:
                                        label_to_indices[d][label_val] = []
                                    label_to_indices[d][label_val].append(j)

                    # Mix
                    for i in range(batch_size):
                        cur_label = labels_train[i].item()  # labels of the current sample
                        # Find all samples in other domains with the same label
                        available_pairs = []
                        for d in label_to_indices:
                            if cur_label in label_to_indices[d]:
                                available_pairs.extend([(d, idx) for idx in label_to_indices[d][cur_label]])

                        if available_pairs:
                            for modality, (data_tensor, all_data_idx) in [
                                ('vib', (datas_vib, 0)),
                                ('cur', (datas_cur, 1)),
                                ('aud', (datas_aud, 2))
                            ]:
                                if random.random() < 0.5:  # 50% possibility
                                    other_d, other_idx = random.choice(available_pairs)
                                    lam = np.random.beta(0.2, 0.2)
                                    if random.random() < 0.5:
                                        lam = 2 - max(lam, 1-lam)
                                    # Mix
                                    if modality == 'vib':
                                        new_datas_vib[i] = lam * data_tensor[i] + (1 - lam) * all_domain_data[other_d][all_data_idx][other_idx]
                                    elif modality == 'cur':
                                        new_datas_cur[i] = lam * data_tensor[i] + (1 - lam) * all_domain_data[other_d][all_data_idx][other_idx]
                                    else:
                                        new_datas_aud[i] = lam * data_tensor[i] + (1 - lam) * all_domain_data[other_d][all_data_idx][other_idx]

                    # feas_vib = new_feas_vib
                    # feas_cur = new_feas_cur
                    # feas_aud = new_feas_aud
                    datas_vib = new_datas_vib
                    datas_cur = new_datas_cur
                    datas_aud = new_datas_aud

                feas_vib, feas_cur, feas_aud = model(datas_vib, datas_cur, datas_aud)
                # modality-level feature decoupling
                feat_dim = feas_vib.size(1) // 2
                vib_inv, vib_spec = feas_vib[:, :feat_dim], feas_vib[:, feat_dim:]
                cur_inv, cur_spec = feas_cur[:, :feat_dim], feas_cur[:, feat_dim:]
                aud_inv, aud_spec = feas_aud[:, :feat_dim], feas_aud[:, feat_dim:]
                
                # modality-level MMD loss
                mmd_loss_vib_cur = mmd_loss(vib_inv, cur_inv)
                mmd_loss_vib_aud = mmd_loss(vib_inv, aud_inv)
                mmd_loss_cur_aud = mmd_loss(cur_inv, aud_inv)
                mmd_total = mmd_loss_vib_cur + mmd_loss_vib_aud + mmd_loss_cur_aud
                
                # modality-level orthogonality constraint
                orth_loss_vib = covariance_loss(vib_inv, vib_spec)
                orth_loss_cur = covariance_loss(cur_inv, cur_spec)
                orth_loss_aud = covariance_loss(aud_inv, aud_spec)
                
                # orthogonality loss between different modalities
                orth_loss_vib_cur = covariance_loss(vib_spec, cur_spec)
                orth_loss_vib_aud = covariance_loss(vib_spec, aud_spec)
                orth_loss_cur_aud = covariance_loss(cur_spec, aud_spec)
                modality_orth_total = orth_loss_vib + orth_loss_cur + orth_loss_aud + \
                                     orth_loss_vib_cur + orth_loss_vib_aud + orth_loss_cur_aud
                
                # feature fusion
                domain_feat = domain_extractor(feas_vib, feas_cur, feas_aud)
                
                # domain-level feature decoupling
                domain_feat_dim = domain_feat.size(1) // 2
                domain_inv, domain_spec = domain_feat[:, :domain_feat_dim], domain_feat[:, domain_feat_dim:]
                
                # store domain features for domain-level loss computation
                domain_features.append((domain_inv, domain_spec))
                
                # clasification using fused features
                outputs_train = classifier(domain_feat)
                cls_loss += loss_fn(outputs_train, labels_train)
                
                batch_preds.append(outputs_train.cpu().data.numpy())
                batch_labels.append(labels_train.cpu().numpy())
            
            # domain-level loss
            if len(domain_features) > 1:
                # MMD loss between domain-invariant features of different domains
                domain_mmd_loss = 0
                for i in range(len(domain_features)):
                    for j in range(i+1, len(domain_features)):
                        domain_mmd_loss += mmd_loss(domain_features[i][0], domain_features[j][0])
                # Orthogonality constraint loss between domain-invariant and domain-specific features within each domain
                # and orthogonality constraint loss between domain-specific features of different domains
                domain_orth_loss = 0
                for i in range(len(domain_features)):
                    # Orthogonality constraint loss between domain-invariant and domain-specific features within the same domain
                    domain_orth_loss += covariance_loss(domain_features[i][0], domain_features[i][1])
                    # orthogonality constraint loss between domain-specific features of different domains
                    for j in range(i+1, len(domain_features)):
                        domain_orth_loss += covariance_loss(domain_features[i][1], domain_features[j][1])
                  # total loss
                total_loss = cls_loss + \
                            opt.lambda_mmd * mmd_total + \
                            opt.lambda_cov * modality_orth_total + \
                            opt.lambda_domain_mmd * domain_mmd_loss + \
                            opt.lambda_domain_cov * domain_orth_loss
            else:
                # if only one domain in the batch, only use modality-level losses
                total_loss = cls_loss + \
                            opt.lambda_mmd * mmd_total + \
                            opt.lambda_cov * modality_orth_total
            
            # backpropagation and optimization step
            optim.zero_grad()
            total_loss.backward()
            optim.step()
            
            train_loss += total_loss
            
            # add batch predictions and labels to the overall lists
            train_preds.extend(batch_preds)
            train_labels.extend(batch_labels)

            pbar.set_postfix({'loss': train_loss.item()})

        lr_scheduler.step()

         # compute accuracy for the epoch
        if train_preds and train_labels:
            train_preds = np.concatenate(train_preds)
            train_labels = np.concatenate(train_labels)
            train_accuracy = compute_accuracy(predictions=train_preds, labels=train_labels)

            print(
                'ite:', ite+1,
                'train_loss:', train_loss.item(),
                'train_accuracy:', train_accuracy)

            # validation
            if (ite+1) % opt.val_every == 0 and ite != 0:
                best_accuracy_val = val_workflow(src_val_dataloader, opt, model, classifier, domain_extractor, ite, best_accuracy_val)


def test_mix(opt, tar_dataset, model, classifier, domain_extractor):
    model.eval()
    classifier.eval()
    domain_extractor.eval()

    tar_dataloader = torch.utils.data.DataLoader(tar_dataset['test'],
                                                 batch_size = opt.batch_size,
                                                #  num_workers=4,
                                                #  pin_memory=True,
                                                 shuffle=False)
    
    test_preds = []
    test_labels = []
    for batch_idx, (inputs, labels) in enumerate(tar_dataloader):
        vibs_test, curs_test, auds_test = inputs['vibration'], inputs['current'], inputs['audio']
        
        vibs_test, curs_test, auds_test, labels = vibs_test.cuda(), curs_test.cuda(), auds_test.cuda(), labels.cuda()
        
        with torch.no_grad():
            feas_vib, feas_cur, feas_aud = model(vibs_test, curs_test, auds_test)
            # combined_feat = torch.cat([feas_vib, feas_cur, feas_aud], dim=1)
            domain_feat = domain_extractor(feas_vib, feas_cur, feas_aud)
            outputs_test = classifier(domain_feat)
            
            predictions = outputs_test.cpu().data.numpy()
            test_preds.append(predictions)
            test_labels.append(labels.cpu().numpy())
            
    predictions = np.concatenate(test_preds)
    labels = np.concatenate(test_labels)

    accuracy = compute_accuracy(predictions=predictions, labels=labels)
    print('----------accuracy test----------:', accuracy)

    log_path = os.path.join(opt.experiment_root, 'test.txt')
    write_log(str('test accuracy:{}'.format(accuracy)), log_path=log_path)
    print('test accuracy:{}'.format(accuracy))
    
    # confusion matrix
    matrix = np.zeros((8, 8))
    for i in range(0, len(labels)):
        matrix[labels[i]][np.argmax(predictions, axis=-1)[i]] += 1
        
    # outfile = os.path.join(opt.experiment_root, 'last_model.pth')
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'classifier_state_dict': classifier.state_dict()
    # }, outfile)

    return accuracy, matrix