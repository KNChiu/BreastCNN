import os
import sys
import wandb
import random
import numpy as np
from PIL import Image
from model.focal_loss import FocalLoss
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model.cnn_cbam import CNN

import wandb
import time
#%%
EPOCH = 100


def fit_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 20, T_mult = 2, eta_min = 0, last_epoch = -1, verbose = False
    )

    loss_func = FocalLoss(2, alpha = torch.tensor([0.4, 0.6]).cuda(), gamma = 4)
    for epoch in range(EPOCH):
        model.train()
        training_loss = 0
        train_correct = 0
        y_true, y_score = [], []

        for idx, (x, y) in enumerate(train_loader):
            output = model(x.to(device))
            loss = loss_func(output, y.to(device))
            training_loss += loss.item()

            train_pred = torch.max(output.data, 1)[1] 
            train_correct += (train_pred == y.to(device)).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        

                
        with torch.no_grad():
            val_loss = 0
            val_correct = 0
            for idx, (x_, y_) in enumerate(val_loader):
                output = model(x_.to(device))
                val_pred = torch.max(output.data, 1)[1] 
                val_correct += (val_pred == y_.to(device)).sum()
                
                loss_ = loss_func(output, y_.to(device))
                val_loss += loss_.item()

        scheduler.step()

        train_Acc = np.round(train_correct.item() / 201, 2)
        val_Acc = np.round(val_correct.item() / 25, 2)


        wandb.log({
                    "Train Loss": training_loss,
                    "Val Loss": val_loss,
                    "Train Acc" : train_Acc,
                    "Val ACC" : val_Acc
                    # "Train Auc" : train_Auc,
                    # "Val Auc" : val_Auc

                   })

        print('Epoch : {}'.format(epoch + 1))
        print('Train Loss : {}  Val Loss : {}'.format(training_loss, val_loss))
        print('Train Acc : {}  Val Acc : {}'.format(train_Acc, val_Acc))
        # print('Train Auc : {}  Val Auc : {}'.format(train_Auc, val_Auc))
        print('=======================================================')



def confusion(y_true, y_pred, mode, logPath):
    confmat = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center', fontsize=10)
    
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Sensitivity = TP / (TP + FN)


    plt.xlabel('Predict', fontsize=10)        
    plt.ylabel('True', fontsize=10)
    
    plt.title(str(mode)+' Accuracy : {:.2f} | Specificity : {:.2f} | Sensitivity : {:.2f}'.format(Accuracy, Specificity, Sensitivity), fontsize=10)
    plt.savefig(logPath+"//"+str(mode)+"_confusion .jpg", bbox_inches='tight')
    plt.close('all')
    # plt.show()

    return Accuracy

def FinalTest(model, test_loader, mode, logPath):
    model.eval()
    correct = 0
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            pred = model(x.to(device))
            y_score.append(pred[0][1].item())
            pred = torch.max(pred.data, 1)[1] 
            correct += (pred == y.to(device)).sum()
            y_true.append(y.item())
            y_pred.append(pred.item())
            
            
    print(str(mode) + '_AUC :', roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    plt.plot(fpr, tpr, label = str(mode) + '_Auc : {:.2f}'.format(roc_auc_score(y_true, y_score)))
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Curve')
    plt.savefig(logPath+"//"+ str(mode) + '_Auc' +".jpg", bbox_inches='tight')
    plt.close('all')
    return y_true, y_pred         
    
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



#%%
if __name__ == '__main__':
    setup_seed(42)

    run = wandb.init(project='Breast_NewCNN', entity='y9760210', reinit=True)

    TRAIN_PATH = r'data\histogram_cc\train'
    VAL_PATH = r'data\histogram_cc\val'
    TEST_PATH = r'data\histogram_cc\test'

    logPath = r"logs//" + str(time.strftime("%m%d_%H%M", time.localtime()))
    if not os.path.isdir(logPath):
        os.mkdir(logPath)

    transform1 = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),])
                                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    transform2 = transforms.Compose([transforms.Resize((640, 640)),
                                    transforms.ToTensor(),])
                                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                                    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU State:', device)

    train = ImageFolder(TRAIN_PATH, transform1)
    val = ImageFolder(VAL_PATH, transform2)
    test = ImageFolder(TEST_PATH, transform2)

    train_loader = DataLoader(train, batch_size = 16, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val, shuffle = True)
    test_loader = DataLoader(test, shuffle = True)
    
    model = CNN().to(device)
    fit_model(model, train_loader, val_loader)
    run.finish()

    val_true, val_pred = FinalTest(model, val_loader, 'val', logPath)
    confusion(val_true, val_pred, 'val', logPath)

    y_true, y_pred = FinalTest(model, test_loader, 'test', logPath)
    confusion(y_true, y_pred, 'test', logPath)


    torch.save(model.state_dict(), './weight/cnn_cbam-418' + '.pth')







