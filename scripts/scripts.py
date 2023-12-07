import torch
from models.EEGSimpleConv import EEGSimpleConv
import numpy as np
import random

import wandb ### uncomment this line to use wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#batch_size = 288



        

def load_data(dict_config):
    """Returns X and Y according to the configuration dict

    Parameters:
    dict_config (dict): configuration dict

    Returns:
    X (list of torch tensors): EEG time series
    Y (list of torch tensors): Associated labels

   """
    dict_config = correct_dict_0(dict_config)
    dict_classes = {'BNCI':4,'Cho':2,'Physionet':4,'Zhou':3,'Weibo':4,'Large':2}
    dict_config['n_classes'] = dict_classes[dict_config['dataset']]
    path=dict_config['path']
    dataset=dict_config['dataset']
    if dict_config['dataset'] not in ['BNCI','Zhou']:
        dict_config['EOG']== False  
    EOG = dict_config['EOG']
    ea = '_EA' if dict_config['EA'] else ''
    s = '_s' if dict_config['session'] else ''
    x_path = path + '/' +dataset+'/X'+ea+s+'.pt'
    if 'online' not in dict_config.keys():
        dict_config['online']=False
    if dict_config['online']==True:
        if dict_config['within']: 
            X = torch.load(path + '/' +dataset+'/X_EA_online_ft.pt')
            Y = torch.load(path + '/' +dataset+'/Y_s.pt')
        else : 
            X = torch.load(path + '/' +dataset+'/X_s.pt')
            Y = torch.load(path + '/' +dataset+'/Y_s.pt')
            assert dict_config['BN'] == False and dict_config['EA'] == False
    else : 
        X = torch.load(x_path)
        Y = torch.load(path + '/' +dataset+'/Y'+s+'.pt')
    if EOG:
        print('warning EOG not supported in this code anymore')
        assert EOG ==False
    if dict_config['model']=='EEGSimpleConv':
        n_chan = X[0][0].shape[1] if dict_config['session'] else X[0].shape[1]
        dict_config['params'].append(n_chan)
        dict_config['params'].append(dict_config['n_classes'])  
        dict_config['params'].append(dict_config['sfreq'])
        if dict_config['reg_subject']:
            n_subjects = len(X)
            dict_config['params'].append(n_subjects)   
    # Check 
    if dict_config['reg_subject']:
        assert len(dict_config['params'])==9
    else  :
        assert len(dict_config['params'])==7   
    
    return X,Y    


def loaders(idx,X,Y,lmso,nsplit,session,reg_subject,within=False,mdl=False):
    """Returns data loaders to train and test for a given split

    Parameters:
    idx (int): id of the split
    X (list of torch tensors): EEG time series
    Y (list of torch tensors): Associated labels
    lmso (bool): True if LMSO, False if LOSO
    nsplit (int): Number of splits
    session (bool): True if session else False
    reg_subject (bool): True if subject-wise regularization else False
    within (bool): True if Within-Subject evaluation else False
    mdl (bool): True if MDL evluation else False
    
    Returns:
    train_loader (torch dataloader)
    test_loader (list of zipped test set)
   """
    n_chan = X[0][0].shape[1]
    if reg_subject:
        Y_subject = [[torch.tensor([i]*XXX.shape[0]) for XXX in XX] for i,XX in enumerate(X)] if session else  [torch.tensor([i]*XX.shape[0]) for i,XX in enumerate(X)] 
    if lmso :
        if session ==False : 
            n_subjects = len(X) 
            inf = (idx * n_subjects) // nsplit
            sup = (n_subjects + idx * n_subjects) // nsplit 
            print(inf,sup)
            train_X = torch.cat(X[:inf] + X[sup:])
            train_Y = torch.cat(Y[:inf] + Y[sup:])
            test_X = X[inf:sup]
            test_Y = Y[inf:sup]
            if reg_subject:#temp
                train_Y_subject = torch.cat(Y_subject[:inf] + Y_subject[sup:]) 
                test_Y_subject = Y_subject[inf:sup]
        if session:
            n_subjects = len(X) 
            inf = (idx * n_subjects) // nsplit
            sup = (n_subjects + idx * n_subjects) // nsplit 
            print(inf,sup)
            X_ = X[:inf] + X[sup:]
            Y_ = Y[:inf] + Y[sup:]
            train_X = torch.cat([item for sublist in X_ for item in sublist])
            train_Y = torch.cat([item for sublist in Y_ for item in sublist])
            test_X = [item for sublist in X[inf:sup] for item in sublist]
            test_Y = [item for sublist in Y[inf:sup] for item in sublist]
            if reg_subject:#temp
                train_Y_subject =torch.cat([item for sublist in Y_subject[:inf] + Y_subject[sup:] for item in sublist])
                test_Y_subject = [item for sublist in Y_subject[inf:sup] for item in sublist]
    else :
        if mdl:
            X_ = X[:idx] + X[idx+1:] + [[X[idx][0]]]
            Y_ = Y[:idx] + Y[idx+1:] + [[Y[idx][0]]]
            train_X = torch.cat([item for sublist in X_ for item in sublist])
            train_Y = torch.cat([item for sublist in Y_ for item in sublist])
            test_X = [X[idx][1]]
            test_Y = [Y[idx][1]]
            if reg_subject:
                Y_subject_ = Y_subject[:idx] + Y_subject[idx+1:]  + [[Y_subject[idx][0]]]
                train_Y_subject = torch.unbind(torch.cat([item for sublist in Y_subject_ for item in sublist]))
                test_Y_subject = [Y_subject[idx][1]]
            
        elif within :
            train_X = X[idx][0]
            test_X = [X[idx][1]]
            train_Y = Y[idx][0]
            test_Y = [Y[idx][1]]
            
        elif within==False and session:
            X_ = X[:idx] + X[idx+1:]
            Y_ = Y[:idx] + Y[idx+1:]
            train_X = torch.cat([item for sublist in X_ for item in sublist])
            train_Y = torch.cat([item for sublist in Y_ for item in sublist])
            test_X = X[idx]
            test_Y = Y[idx]
            if reg_subject:
                Y_subject_ = Y_subject[:idx] + Y_subject[idx+1:]
                train_Y_subject = torch.unbind(torch.cat([item for sublist in Y_subject_ for item in sublist]))
                test_Y_subject = Y_subject[idx]
        else:
            train_X = torch.cat(X[:idx] + X[idx+1:])
            train_Y = torch.cat(Y[:idx] + Y[idx+1:])
            test_X = [X[idx]]
            test_Y = [Y[idx]]
            if reg_subject:
                train_Y_subject = torch.unbind(torch.cat(Y_subject[:idx] + Y_subject[idx+1:])) #temp
                test_Y_subject = [Y_subject[idx]]
    
    if within:
        batch_size = 16
    else :
        batch_size = 288
    mean = train_X.transpose(1,2).reshape(-1, n_chan).mean(dim = 0)
    std = train_X.transpose(1,2).reshape(-1, n_chan).std(dim = 0)
    train_X = (train_X - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2)

    train_X = torch.unbind(train_X)
    train_Y = torch.unbind(train_Y)
    train_data = list(zip(train_X, train_Y,train_Y_subject)) if reg_subject else list(zip(train_X, train_Y))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = 8)
    test_X = [(XX - mean.unsqueeze(0).unsqueeze(2)) / std.unsqueeze(0).unsqueeze(2) for XX in test_X]
    test_loader = list(zip(test_X, test_Y,test_Y_subject)) if reg_subject else list(zip(test_X, test_Y))
    return train_loader, test_loader

    
def train(epoch, model, criterion, optimizer, train_loader, mixup = False,T=0.1,preload_reg=False):
    """Train the model for one epoch

    Parameters:
    epoch (int): epoch number
    model (torch model)
    criterion (torch loss)
    optimizer (torch optimizer)
    train_loader (torch dataloader)
    mixup (bool): True if use mixup else False
    T (float): Temperature to ponderate the subject-wise regularization term in the loss
    preload_reg (bool): True if the pretrained model was trained using subject-wise reguralization
    
    Returns:
    Train accuracy
   """
    losses, scores = [], []
    cont = True
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        reg_subject =True if len(batch_data)==3 else False
        if reg_subject:
            data,target,target_subject = batch_data
            data,target,target_subject = data.to(device), target.to(device),target_subject.to(device)
        else : 
            data,target = batch_data
            data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if mixup:
            mm = random.random()
            perm = torch.randperm(data.shape[0])
            if reg_subject or preload_reg:
                output,output_subject = model(mm * data + (1 - mm) * data[perm])
            else :
                output = model(mm * data + (1 - mm) * data[perm]) 
        else:
            if reg_subject or preload_reg:
                output,output_subject = model(data)
            else : 
                output = model(data)
        decisions = torch.argmax(output, dim = 1)
        scores.append((decisions == target).float().mean().item())
        if mixup:
            loss = mm * criterion(output, target) + (1 - mm) * criterion(output, target[perm])
        else:
            loss = criterion(output, target)
        if reg_subject:
            #loss2 = criterion(output_subject,target_subject) #
            loss2 = mm * criterion(output_subject, target_subject) + (1 - mm) * criterion(output_subject, target_subject[perm])  
            loss = (loss + T*loss2)/2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print("\r{:3d} {:3.3f} {:3.3f} ".format(epoch + 1, np.mean(losses), np.mean(scores)), end='')
    return np.mean(scores)

def test(epoch, model, test_loader,bn, confusions = False,preload_reg=False):
    """Test the model for one epoch

    Parameters:
    epoch (int): epoch number
    model (torch model)
    test_loader (list of zipped test set)
    bn (bool): True if use BN trick else False
    confusions (bool): True if print confusion matrix else False
    preload_reg (bool): True if the pretrained model was trained using subject-wise reguralization
    
    Returns:
    Test accuracy
   """
    if confusions:
        confs = torch.zeros((4,4))
    score, count = 0, 0
    if bn :
        model.train()
    else : 
        model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            reg_subject =True if len(batch_data)==3 else False
            if reg_subject:
                (data, target,_) = batch_data
            else : 
                (data, target) = batch_data
            data, target = data.to(device), target.to(device)
            if reg_subject or preload_reg:
                output,_ = model(data)
            else : 
                output = model(data)
            decisions = torch.argmax(output, dim = 1)
            if confusions:
                for j in range(4):
                    for k in range(4):
                        confs[j][k] += (decisions[torch.where(target == j)[0]] == k).int().sum().item()
            score += (decisions == target).int().sum().item()
            count += target.shape[0]
    print("\r{:3d} test: {:.3f} ".format(epoch, score / count), end = '')
    if confusions:
        print(confs)
    return (score / count)
    

def train_test(params, dict_config,X,Y):
    """Train and test the model according to the configuration dict

    Parameters:
    params (list): model paramaters
    dict_config (dict): configuration dict
    X (list of torch tensors): EEG time series
    Y (list of torch tensors): Associated labels
    
    Returns:
    Test accuracy
   """
    dict_config['params'] = params
    dict_config,params_model = correct_dict_1(dict_config,params)
    if dict_config['preload_reg']==True:
        params_model = params + [len(Y)]
    if dict_config['use_wandb']:
        wandb.init(project="simpleconv_c", entity='brain-imt',config = dict_config,settings=wandb.Settings(start_method='fork'))
    
    
    model = instanciate_model(dict_config['model'],params_model)
    number_params =  np.sum([m.numel() for m in model.parameters()])
    print(params,number_params, "params")
    runs = dict_config['runs']
    n_split = 9 if dict_config['lmso'] else len(X)
    scores = []
    for idx_ in range(n_split):
        print("Split:", idx_)
        train_loader, test_loader = loaders(idx_,X,Y,dict_config['lmso'],n_split,dict_config['session'],dict_config['reg_subject'],within=dict_config['within'],mdl=dict_config['mdl'])
        criterion = torch.nn.CrossEntropyLoss()
        for n_run in range(runs):
            if dict_config['load_model']:
                model = instanciate_model(dict_config['model'],params_model).to(device)
                checkpoint = torch.load(dict_config['load_model_path']+'/model_'+str(idx_)+'_'+str(n_run)+'.pt')
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer = torch.optim.Adam(model.parameters())
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                model = instanciate_model(dict_config['model'],params_model).to(device)
                optimizer = torch.optim.Adam(model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = (4*dict_config['n_epochs'])//5, gamma = 0.1)
            for epoch in range(dict_config['n_epochs']):
                train_acc = train(epoch, model, criterion, optimizer, train_loader, mixup = dict_config['mixup'],T=dict_config['T'],preload_reg=dict_config['preload_reg'])
                if epoch ==dict_config['n_epochs'] -1 : #%2==0
                    if dict_config['save_model']:
                        torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},dict_config['save_model_path']+'/model_w_'+str(idx_)+'_'+str(n_run)+'.pt')
                    score = test(epoch, model, test_loader,dict_config['BN'],preload_reg=dict_config['preload_reg'])
                scheduler.step()
            scores.append(score)
            
            if n_run == runs - 1:
                print(" average: {:.3f}".format(np.mean(scores[-runs:])))
            else:
                print()

    std_sub = np.array(scores).reshape(n_split,runs).mean(1).std()
    std = np.array(scores).reshape(n_split,runs).T.mean(1).std()
    print("{:.3f}".format(np.mean(scores)))
    if dict_config['use_wandb']:
        wandb.log({'test_acc':np.mean(scores),'scores':scores,'number_params':number_params,'std':std,'std_sub':std_sub})
        wandb.finish()


    return np.mean(scores)


def instanciate_model(model_name, params_model):
    """Initialize the model

    Parameters:
    model_name (string): Name of the model (only valid for EEGSimpleConv)
    params_model (list): model paramaters
    
    Returns:
    torch model
   """
    if model_name == 'EEGSimpleConv':
        return EEGSimpleConv(*params_model)


    
### Two functions to ensure there are no inconsistencies in the use of the dictionnary
    
def correct_dict_0(dict_config):
    if dict_config['dataset'] not in ['BNCI','Zhou','Large']:
        dict_config['lmso']   =True
        dict_config['EOG']    =False
        dict_config['session']=False
        print('no EOG or session in Physionet or Cho')
    else :
        dict_config['lmso']   =False
        dict_config['sfreq'] = 250
    if dict_config['dataset']=='Cho':
        dict_config['sfreq'] = 512
    if dict_config['dataset']=='Physionet':
        dict_config['sfreq'] = 160
    print('Sampling Freq: '+str(dict_config['sfreq']))
    
    if dict_config['within']==True or dict_config['mdl']=='True':
        assert dict_config['dataset']=='BNCI'
    if dict_config['evaluation']=='cross':
        assert dict_config['within']==False
        assert dict_config['mdl']==False
    if dict_config['within']==True:
        assert dict_config['reg_subject']==False
    return dict_config


def correct_dict_1(dict_config,params):
    if dict_config['reg_subject']:
        dict_config['T'] = params[0]
        params_model = params[1:]
        assert len(params_model)==8 and len(params)==9
    else : 
        dict_config['T'] = None
    if len(params)==9:
        assert dict_config['reg_subject']==True
    else : 
        params_model = params
        assert len(params_model)==7
    return dict_config,params_model