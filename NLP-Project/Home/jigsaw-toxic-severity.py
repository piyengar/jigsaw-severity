#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os  # operating system library
import gc  # Garbage Collector - module provides the ability to disable the collector, tune the collection frequency, and set debugging options
import copy  # The assignment operation does not copy the object, it only creates a reference to the object. 

import time  # time library


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  # DataLoader and other utility functions for convenience
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split  # Stratified K-Folds cross-validator.



# Util
from tqdm import tqdm
from collections import defaultdict  # Usually, a Python dictionary throws a KeyError if you try to get an item with a key that is not currently in the dictionary. 


# Huggingface imports
from transformers import AutoTokenizer, AutoModel, AdamW  

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Config
# Add configurations to this dict

# In[2]:


CONFIG = {
    "seed": 666,
    "epochs": 1000,
    "model_name": "distilbert-base-uncased",
    "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "max_length": 128,
    "n_accumulate": 1,       # Gradient accumulation , set > 1 to use
    "n_fold": 10,             # The number of folds to split the data in, pick one for validation. We are only dpoing this for val split for now
    "learning_rate": 1e-5,
    "weight_decay": 1e-6,
    "train_batch_size": 16,
    "valid_batch_size": 64,
    "is_dev_run": False,
    "margin": 0.0
}
CONFIG['model_path'] = f"{CONFIG['model_name']}"
CONFIG['tokenizer'] = AutoTokenizer.from_pretrained(CONFIG['model_path'])


# In[3]:


# Sets the seed of the entire notebook so results are the same every time we run.
# This is for REPRODUCIBILITY
def set_seed(seed=42): 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])


# ### Datasets
# Load datasets into dataframes. We can use multiple datasets in addition to the data provided to this competetion. We use:
# - jigsaw-toxic-severity-rating (val & sub)
# - jigsaw-toxic-comment-classification (train)
# - ...Add more here

# #### jigsaw-toxic-comment-classification

# In[4]:


df_cc = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
print(df_cc.shape)


# In[5]:


df_cc.columns


# Combine the different toxicity flags into a single value

# In[6]:


# apply a weight to each kind of toxicity, can we learn this weight??
tox_weight = np.array([1,1,1,1,1,1])
# multiply the flags with the weight

df_cc['y'] = ((df_cc[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']] * tox_weight).sum(axis=1) )
Y = np.array(df_cc[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])
df_cc['y'] = df_cc['y'] / np.sum(tox_weight) # Normalize
print(df_cc[df_cc.y > 0][df_cc.y < 0.8].sample(5, random_state=4))
print(f"max = {df_cc.y.max()}, min = {df_cc.y.min()}")


# In[7]:


Y.shape


# Retain only text and y

# In[8]:


df_cc = df_cc.rename(columns={'comment_text':'text'})
df_cc.head()


# ### Kfold split for randomly selecting validation set
# We only pick the last fold for val right now, but we can explore kfold ensemlbing later as well

# In[9]:


skf = KFold(n_splits=CONFIG['n_fold'], shuffle=True, random_state=CONFIG['seed'])  # set the parameters for splitting our dataframe into data for training and testing

splits = skf.split(df_cc)
for fold, (_, val_) in enumerate(splits):  # dataframe splitting
    df_cc.loc[val_ , "kfold"] = int(fold)
    
df_cc["kfold"] = df_cc["kfold"].astype(int)  # add one more column of folder number to the original dataframe
df_cc.head()  # display the first 5 rows of the dataframe table


# #### jigsaw-toxic-severity-rating
# Validation and Submission datasets

# In[10]:


df_val = pd.read_csv("../input/jigsaw-toxic-severity-rating/validation_data.csv")
print(df_val.shape)
df_val.head()


# ### Kfold split for randomly selecting validation set
# We only pick the last fold for val right now, but we can explore kfold ensemlbing later as well

# In[11]:


skf = KFold(n_splits=CONFIG['n_fold'], shuffle=True, random_state=CONFIG['seed'])  # set the parameters for splitting our dataframe into data for training and testing

splits = skf.split(df_val)
for fold, (_, val_) in enumerate(splits):  # dataframe splitting
    df_val.loc[val_ , "kfold"] = int(fold)
    
df_val["kfold"] = df_val["kfold"].astype(int)  # add one more column of folder number to the original dataframe
df_val.head()  # display the first 5 rows of the dataframe table


# In[12]:


df_sub = pd.read_csv("../input/jigsaw-toxic-severity-rating/comments_to_score.csv")
print(df_sub.shape)
df_sub.head()


# ## Dataset class

# In[13]:


class SeverityDataset(Dataset):
    def __init__(self, df, tokenizer: AutoTokenizer, max_length, load_target=True):
        self.load_target = load_target
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df['text'].values
        if self.load_target:
            self.target = np.array(df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
                        text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.max_len,
                        padding='max_length'
                    )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        data = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
        if self.load_target:
            target = torch.tensor(self.target[index], dtype=torch.float)
            data['target'] = target
        return data

### TEST
temp_ds = SeverityDataset(df_cc, CONFIG['tokenizer'], CONFIG['max_length'])
print(temp_ds[0])

# del temp_ds


# #### Test dataset class

# In[14]:


class ContrastiveDataset(Dataset):
    def __init__(self, df, tokenizer: AutoTokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.more_toxic = df['more_toxic'].values
        self.less_toxic = df['less_toxic'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        more_toxic_text = self.more_toxic[index]
        more_toxic_inputs = self.tokenizer.encode_plus(
                        more_toxic_text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.max_len,
                        padding='max_length'
                    )
        
        more_toxic_ids = more_toxic_inputs['input_ids']
        more_toxic_mask = more_toxic_inputs['attention_mask'] 
        
        less_toxic_text = self.less_toxic[index]
        less_toxic_inputs = self.tokenizer.encode_plus(
                        less_toxic_text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.max_len,
                        padding='max_length'
                    )
        
        less_toxic_ids = less_toxic_inputs['input_ids']
        less_toxic_mask = less_toxic_inputs['attention_mask'] 
        # the target here is the difference in toxicity. Since our severity range is 1 we set the difference between more and less toxic to 1
        target = 1
        return {
            'more_toxic_ids': torch.tensor(more_toxic_ids, dtype=torch.long),
            'more_toxic_mask': torch.tensor(more_toxic_mask, dtype=torch.long),
            'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
            'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }
### TEST
temp_ds = ContrastiveDataset(df_val, CONFIG['tokenizer'], CONFIG['max_length'])
print(temp_ds[0])

del temp_ds


# ### Define model for pretraining

# In[15]:


class SeverityModel(nn.Module):  
    def __init__(self, model_path):  # initialization of the class at the input of the dataframe, tokenizer, max_length
        # set the class attributes
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, 6)
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.drop(out[0])
        outputs = self.fc(out)
        
        return outputs  # returns the obtained values


# In[50]:


class JigsawModel(nn.Module):  
    def __init__(self, model_path, mode='severity'):  # initialization of the class at the input of the dataframe, tokenizer, max_length
        # set the class attributes
        super(JigsawModel, self).__init__()
        self.mode = mode
        self.model = AutoModel.from_pretrained(model_path)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(768, 300)
        self.fc2 = nn.Linear(300, 1)
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.drop(out[0])
        #out = out.mean(1)
        #print(out.shape)
        out = self.fc(out).relu()
        #print(out.shape)
        if self.mode == 'contrastive':
            out = self.fc2(out)
        
        return out  # returns the obtained values

class MultiModel(nn.Module):  
    def __init__(self, model_path, mode='severity'):  # initialization of the class at the input of the dataframe, tokenizer, max_length
        # set the class attributes
        super(MultiModel, self).__init__()
        self.mode = mode
        self.model = AutoModel.from_pretrained(model_path)
        self.drop = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(768, 300)
        self.fc = nn.Linear(300, 6)
        self.fc2 = nn.Linear(300, 1)
        
    def forward(self, ids, mask):        
        out = self.model(input_ids=ids,attention_mask=mask,
                         output_hidden_states=False)
        out = self.drop(out[0])
        out = self.fc3(out.relu()).relu()
        return (self.fc(out), self.fc2(out))  # returns the obtained values


# ## Severity
# ## Train
# Here we train on the classification dataset as we already have a severity score as target.

# In[17]:


def train_severity_one_epoch(model, criterion, optimizer, scheduler, dataloader, device, epoch):  
    # one epoch training function
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for step, data in bar:
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)

        outputs = model(ids, mask)
        
        loss = criterion(outputs, targets)
        loss = loss / CONFIG['n_accumulate']
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
        if CONFIG['is_dev_run'] and step > 5:
            # Break after one step
            break
    gc.collect()
    
    return epoch_loss  # returns the result of the training function for one epoch



# ### Validate

# In[18]:


@torch.no_grad()
def valid_severity_one_epoch(model, criterion, optimizer, dataloader, device, epoch):  # one epoch check function
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)

        outputs = model(ids, mask)
        
        loss = criterion(outputs, targets)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
        if CONFIG['is_dev_run'] and step > 5:
            # Break after one step
            break
    
    gc.collect()
    
    return epoch_loss  # returns the result of the check function for one epoch


# ## Contrastive

# ### Train
# We train on the competetion dataset that only has a comparision. So we need to use contrastive loss 

# In[49]:


def train_contrastive_one_epoch(model, criterion, optimizer, scheduler, dataloader, device, epoch):  
    # one epoch training function
    model.train()
    
    dataset_size = 0
    running_loss = 0.0

    epoch_loss = 0.0
    running_acc = 0.0
    epoch_acc = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    
    for step, data in bar:
        #print(step)
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = more_toxic_ids.size(0)
        dataset_size += batch_size
        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        
        loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)
        loss = loss / CONFIG['n_accumulate']
        loss.backward()
        
        
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        
        epoch_loss = running_loss / dataset_size
        
        


        running_acc += (more_toxic_outputs > less_toxic_outputs).sum(1).ge(64).sum()
        #.sum().item()
#         print('running_acc, dataset_size', running_acc, dataset_size)
        epoch_acc = running_acc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'],
                       Train_acc=epoch_acc)
        if CONFIG['is_dev_run'] and step > 5:
            # Break after one step
            break
    gc.collect()
    
    return epoch_loss  # returns the result of the training function for one epoch

def train_multi_one_epoch(model, criterion, optimizer, scheduler, dataloader, device, epoch):  
    # one epoch training function
    sev_criterion, con_criterion = criterion
    sev_dataloader, con_dataloader = dataloader
    
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0.0
    
    length = min(len(sev_dataloader), len(con_dataloader))
    
    bar = tqdm(range(length // CONFIG["train_batch_size"]))
    
    sev_enum = enumerate(sev_dataloader)
    con_enum = enumerate(con_dataloader)
    
    accs = []
    
    
    for step in bar:
        try:
            _, data = next(sev_enum)

        except:
            break
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
       
        try:
            _, data = next(con_enum)
        except:
            break
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
        con_targets = data['target'].to(device, dtype=torch.long)        
        
        
        batch_size = ids.size(0)

        _,outputs_sev = model(ids, mask)
        
        loss_sev = sev_criterion(outputs_sev, targets.sum(-1).unsqueeze(-1))
        
        _, more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        _, less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        
        loss_con = con_criterion(more_toxic_outputs, less_toxic_outputs, con_targets)
        
        loss = loss_sev + loss_con
        
        acc = (more_toxic_outputs > less_toxic_outputs).int().sum(1).ge(64).float().mean()       
        
        loss = loss / CONFIG['n_accumulate']

        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'],acc=acc)
        if CONFIG['is_dev_run'] and step > 5:
            # Break after one step
            break
    gc.collect()
    
    return epoch_loss  # returns the result of the training function for one epoch
    
# ### Validate 

# In[40]:


@torch.no_grad()
def valid_contrastive_one_epoch(model, criterion, optimizer, dataloader, device, epoch, multi=False):  # one epoch check function
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0.0
    
    running_acc = 0.0
    epoch_acc = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
        more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
        less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
        less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = targets.size(0)

        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
        
        if isinstance(model, MultiModel):
            more_toxic_outputs=more_toxic_outputs[1]
            less_toxic_outputs=less_toxic_outputs[1]
            
             
        loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        running_acc += (more_toxic_outputs > less_toxic_outputs).sum(1).ge(64).sum()
        epoch_acc = running_acc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'],
                       Valid_acc=epoch_acc)   
        if CONFIG['is_dev_run'] and step > 5:
            # Break after one step
            break
    gc.collect()
    
    return epoch_loss  # returns the result of the check function for one epoch


# ### Criterion

# In[41]:


def criterion_severity(outputs, targets):  # Creates a criterion that measures the loss
    outputs = nn.functional.sigmoid(outputs) # need to constrain the outputs to [0, 1]
    
    return nn.BCELoss()(outputs.view(-1), targets.float())


# In[42]:


alpha = 0.25
gamma = 2
def criterion_focal(outputs, targets):
    targets = targets.unsqueeze(-1).repeat(1, 128, 1)
    #print(outputs.shape, targets.shape)
    return nn.MSELoss()(outputs.exp(), targets.exp())
    BCE_loss = nn.BCELoss(reduction='none')(outputs, targets.float())
    pt = torch.exp(-BCE_loss) # prevents nans when probability 0
    focal_loss = alpha * (1-pt)**gamma * BCE_loss
    return focal_loss.mean()


# In[43]:


contrast_more_tox, contrast_less_tox = 0, 0
def criterion_contrastive(outputs1, outputs2, targets):  # Creates a criterion that measures the loss
    

    outputs1 = outputs1.view(outputs1.size(0), -1).sigmoid() # need to constrain the outputs to [0, 1]
    outputs2 = outputs2.view(outputs2.size(0), -1).sigmoid() # need to constrain the outputs to [0, 1]
    return nn.BCELoss()(outputs1, torch.ones_like(outputs1)) + nn.BCELoss()(outputs2, torch.zeros_like(outputs2))    
    #targets = targets.reshape(targets.size(0), -1).repeat(1, outputs1.size(1))
#     print(outputs1, outputs2)
    #return nn.MarginRankingLoss(margin=CONFIG['margin'])(outputs1, outputs2, targets)




# ## Run Training

# In[44]:


def run_training(model, optimizer, scheduler, device, num_epochs, train_loader, valid_loader, mode):  # general training function
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    if CONFIG['is_dev_run']:
        num_epochs = 1
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    if mode == 'severity':
        train_fn = train_severity_one_epoch
        valid_fn = valid_severity_one_epoch
        criterion = criterion_focal
    elif mode == 'contrastive':
        train_fn = train_contrastive_one_epoch
        valid_fn = valid_contrastive_one_epoch
        criterion = criterion_contrastive
    elif mode == 'multi':
        train_fn = train_multi_one_epoch
        valid_fn = valid_contrastive_one_epoch
        criterion = (criterion_focal, criterion_contrastive)
    bar = range(1, num_epochs + 1)
    for epoch in bar: 
        #bar.set_description(f"Epoch-{epoch}")
        gc.collect()
        train_epoch_loss = train_fn(model, criterion, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss = valid_fn(model, criterion_contrastive, optimizer, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        
        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"Loss.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


# ### Dataloaders

# In[45]:


def prepare_loaders(df, fold, mode):
    if mode == 'multi':
        df_sev, df_con = df
        df_sev = df_sev[df_sev.kfold != fold].reset_index(drop=True)
        df_con_train = df_con[df_con.kfold != fold].reset_index(drop=True)
        df_valid = df_con[df_con.kfold == fold].reset_index(drop=True)
        sev_train_dataset = SeverityDataset(df_sev, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])
        con_train_dataset = ContrastiveDataset(df_con_train, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])
        valid_dataset = ContrastiveDataset(df_valid, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])
        
        sev_loader = DataLoader(sev_train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=1, shuffle=True, pin_memory=True, drop_last=True)
        con_loader = DataLoader(con_train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=1, shuffle=True, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=1, shuffle=False, pin_memory=True)
        return ((sev_loader, con_loader), valid_loader)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    if mode == 'severity':
        ds_class = SeverityDataset
    elif mode == 'contrastive':
        ds_class = ContrastiveDataset
    train_dataset = ds_class(df_train, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])
    valid_dataset = ds_class(df_valid, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader


# ## Scheduler

# In[46]:


def fetch_scheduler(optimizer):
    sch_name = CONFIG.get('scheduler', None)
    if sch_name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif sch_name == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif sch_name == None:
        return None
        
    return scheduler


# ### Start training Severity

# In[47]:


def run_training_for_mode(mode, model=None):
    if mode == 'severity':
        df = df_cc
    elif mode == 'contrastive':
        df = df_val
    elif mode == 'multi':
        df = (df_cc, df_val)
    train_loader, valid_loader = prepare_loaders(df, CONFIG['n_fold'] - 1, mode)
    model = model or MultiModel(CONFIG['model_path'])
    model.mode = mode
    model.to(CONFIG['device'])

    # Define Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer)

    model, history = run_training(model, optimizer, scheduler,
                                  device=CONFIG['device'],
                                  num_epochs=CONFIG['epochs'],
                                  train_loader=train_loader, 
                                  valid_loader=valid_loader,
                                  mode=mode)


    del train_loader, valid_loader
    _ = gc.collect()
    
    return model, history


# ### Start training
# The training is run one by one on the modes array and the model from each step is fed to the next one.
# 
# The different possible options are:
# - mode = contrastive
# - mode = severity
# - mode = severity -> contrastive
# - mode = contrastive -> severity

# In[48]:


modes = [
    'multi',
#     'multi',
]
model = None
for mode in modes:
    model, history = run_training_for_mode(mode, model)
    # TODO need to display history in some way
modes_str = "_".join(modes)
PATH = f"Model-{modes_str}-{time.time()}.bin"
torch.save(model.state_dict(), PATH)


# In[ ]:





# In[ ]:




