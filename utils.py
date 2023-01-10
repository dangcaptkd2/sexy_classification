import torch 

import copy
import time 
from tqdm import tqdm 
import mlflow 
import numpy as np 
import matplotlib.pyplot as plt 
import itertools
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', export_as='confusion_matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True labels', fontsize=14)
    plt.xlabel('Predicted labels', fontsize=14)

    plt.savefig(f'/quyennt/sexy/images/{export_as}.png', bbox_inches='tight');

def train_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, cfg):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10**6

    for epoch in tqdm(range(cfg['num_epochs']), leave=False, position=0):
        print(f"Epoch {epoch}/{cfg['num_epochs'] - 1}")
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for step, (inputs, labels) in enumerate(tqdm(dataloaders[phase], desc=phase, leave=False, position=0)):
                inputs = inputs.to(cfg['device'])
                labels = labels.to(cfg['device'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f">> Epoch: {epoch} || {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch_loss < best_loss:
                    best_loss = epoch_loss

            if cfg['use_mlflow']:
                    if phase=='train':
                        mlflow.log_metric("Train/Accuracy", epoch_acc)
                        mlflow.log_metric("Train/Loss", epoch_loss)
                        mlflow.log_metric("lr", optimizer.param_groups[0]['lr'])
                    if phase=='val':
                        mlflow.log_metric("Val/Accuracy", epoch_acc)
                        mlflow.log_metric("Val/Loss", epoch_loss)


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} || Best val Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, round(float(best_acc), 3)

def evaluate_model(model, dataloaders, cfg):
    since = time.time()

    model.eval()
    for phase in dataloaders.keys():
        y, y_hat = [], []
   
        for step, (inputs, labels) in enumerate(tqdm(dataloaders[phase], desc=phase, leave=False, position=0)):
            inputs = inputs.to(cfg['device'])
            labels = labels.to(cfg['device'])

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
        
            y.extend(labels.data.cpu().detach().numpy())
            y_hat.extend(preds.cpu().detach().numpy())
        
        # print(">> f1 score (binary)", f1_score(y, y_hat, average='binary'))
        print(">> f1 score (micro)", f1_score(y, y_hat, average='micro'))
        print(">> f1 score (macro)", f1_score(y, y_hat, average='macro'))
        print(">> accuracy score", accuracy_score(y, y_hat))

        cm = confusion_matrix(y, y_hat)
        classes = ['bikini', 'neural', 'nude']
        plot_confusion_matrix(cm, 
                                classes, 
                                normalize=True, 
                                title=f"{cfg['name']}-{cfg['data_name']}-{phase}", 
                                export_as=f"{cfg['name']}_{cfg['data_name']}_{phase}")
        print(f"Done save confusion matrix at: /quyennt/sexy/images/{cfg['name']}_{cfg['data_name']}_{phase}.png")
        plt.show()

def train_student_model(model, dataloaders, criterion, optimizer, scheduler, dataset_sizes, cfg):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10**6

    for epoch in tqdm(range(cfg['num_epochs']), leave=False, position=0):
        print(f"Epoch {epoch}/{cfg['num_epochs'] - 1}")
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for step, (x_teacher, y_teacher, x_student, y_student) in enumerate(tqdm(dataloaders[phase], desc=phase, leave=False, position=0)):
                x_teacher = x_teacher.to(cfg['device'])
                y_teacher = y_teacher.to(cfg['device'])

                x_student = x_student.to(cfg['device'])
                y_student = y_student.to(cfg['device'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_t = model(x_teacher)
                    _, preds_t = torch.max(outputs_t, 1)

                    outputs_s = model(x_student)
                    _, preds_s = torch.max(outputs_s, 1)

                    loss_t = criterion(outputs_t, y_teacher)
                    loss_s = criterion(outputs_s, y_student)
                    loss = loss_t + loss_s

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * x_teacher.size(0)
                running_corrects += torch.sum(preds_t == y_teacher.data)
                running_corrects += torch.sum(preds_s == y_student.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / (dataset_sizes[phase]*2)

            print(f">> Epoch: {epoch} || {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch_loss < best_loss:
                    best_loss = epoch_loss

            if cfg['use_mlflow']:
                    if phase=='train':
                        mlflow.log_metric("Train/Accuracy", epoch_acc)
                        mlflow.log_metric("Train/Loss", epoch_loss)
                        mlflow.log_metric("lr", optimizer.param_groups[0]['lr'])
                    if phase=='val':
                        mlflow.log_metric("Val/Accuracy", epoch_acc)
                        mlflow.log_metric("Val/Loss", epoch_loss)


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} || Best val Loss: {best_loss:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, round(float(best_acc), 3)
