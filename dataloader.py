import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PianoRollDataset(Dataset):
  def __init__(self, data_path = './', 
               dataset_fs = 30, model_fs = 20, 
               ons_value = 2, sus_value = 1, 
               padding_value = -99,
               source_length = 512,
               use_transposition = False,
               preload = True, 
               device = torch.device('cpu'), dtype = torch.float32):
    '''
    description: Dataloader for piano rolls, to be used with a Transformer Encoder model.
    Features optional transposition and downsampling in time.

    parameter: data_path, path to the piano rolls (pytorch tensors of dim: time x pitch)
    parameter: fs, sampling frequency. If fs < 30, the piano rolls will be downsampled in time.
    parameter: ons_value, value of note onsets in the piano roll
    parameter: sus_value, value of note sustains in the piano roll
    parameter: padding_value, used to pad sequences of different lengths in batches
    parameter: source_length, length of the input sequence
    parameter: use_transposition, if True, the piano rolls will be randomly transposed in a random key
    parameter: preload, if True, the entire dataset will be loaded in memory
    parameter: device
    parameter: dtype
    '''
    
    ### class variables
    self.data_path = data_path
    self.dataset_fs = dataset_fs
    self.fs = model_fs
    self.ons_value = ons_value
    self.sus_value = sus_value
    self.padding_value = padding_value
    self.preload = preload
    self.device = device
    self.dtype = dtype
    self.source_length = source_length
    self.use_transposition = use_transposition
    
    #load file list
    self.files = []
    files = os.listdir(data_path)
    for file in files:
      if file.endswith('.pt'):
        self.files.append(file)

    #preload data
    if preload == True:
      self.loaded_data = dict()
      i_seq = 0
      for idx in range(len(self.files)):
        out_list, name_list = self.load_item(self.files[idx])
        for out in out_list:
          self.loaded_data[i_seq] = (out, name_list[0])
          i_seq +=1

  ###class methods
  def __len__(self):
    '''
    parameter:
    returns: length of the entire dataset
    '''
    if self.preload == True:
      length = len(self.loaded_data)
    else:
      length = len(self.files)
    return length

  def __getitem__(self, idx):
    '''
    parameter: index of an item (int)
    returns: the item, and the file_name of that item
    '''
    if torch.is_tensor(idx):
        idx = idx.tolist()

    if self.preload == False:
      out_list, name_list = self.load_item(self.files[idx])
      out, file_name = out_list[0], name_list[0]
    else:
      (out, file_name) = self.loaded_data[idx]

    #transpose in random key
    if self.use_transposition == True:
      seq_len = out.shape[0]
      offset = np.random.randint(-6, 7)
      if offset < 0:
        out = torch.cat((torch.zeros(seq_len,np.abs(offset)),out[:,:offset]),dim = 1)
      elif offset > 0:
        out = torch.cat((out[:,offset:],torch.zeros(seq_len,offset)),dim = 1)
    return out, file_name

  def load_item(self, file_name):
    '''
    parameter: file_name of an item to load
    returns: a list of items and a list of file_names. 
            If time_max is None, the lists contain only one item and its name. 
            If time_max is not None, the lists may contain several chunked items, and several instances of the file_name
    '''
    out_list = []
    name_list = []
    pr = torch.load(self.data_path + file_name, map_location=torch.device(self.device))

    out = torch.zeros(pr.shape)
    out[pr == 1] = self.sus_value
    out[pr == 2] = self.ons_value

    #downsampling in time
    if self.fs < self.dataset_fs:
      chunk_rows = np.linspace(0,out.shape[0],np.round(self.fs*out.shape[0]/30).astype(int) +1) #get downsample time indices
      chunk_rows = np.round(chunk_rows).astype(int) #get nearest integer
      chunk_rows = np.unique(chunk_rows) #remove duplicate indices
      down_sampled = torch.zeros((chunk_rows.shape[0]-1,out.shape[1]), device = out.device, dtype = out.dtype)
      for i0 in range(len(chunk_rows)-1):
        max, _ = torch.max(out[chunk_rows[i0]:chunk_rows[i0+1],:],dim = 0)
        down_sampled[i0,:] = max
      out = down_sampled

    #chunk time axis
    if self.source_length is None:
      out_list.append(out)
      name_list.append(file_name)
      return out_list, name_list
    else:
      chunk_size = self.source_length + 1
      last_t = out.shape[0] // chunk_size
      for t in range(last_t):
        out_part = out[t*chunk_size:(t+1)*chunk_size,:]
        out_list.append(out_part)
        name_list.append(file_name + '_chunk_' + str(t))
      out_list.append(out[-chunk_size:,:])
      name_list.append(file_name)
      return out_list, name_list

  def getitem_byname(self,file_name): #used to retrieve a specific item without relying on its index.
    '''
    parameter: file_name of an item to load
    returns: the item and its name
    '''
    out_list, name_list = self.load_item(file_name)
    out, file_name = out_list[0], name_list[0]
    return out, file_name
  
  def collate_batch(self, batch): #method used to preprocess batches of items (pad and concatenate)
    '''
    parameter: batch, a list of (piano_roll, file_name) tuples. These items will be concatenated in a single batch.
    returns: a batch of samples (used by the model to make predictions) and a batch of targets (used to check the predictions)
    '''
    sample_list, target_list = [], []

    for (pr,_) in batch:
        sample = pr.clone()[:-1, :]
        target = pr.clone()[1:, :]

        #remove sustain values in targets
        target[target == self.sus_value] = 0
        target[target == self.ons_value] = 1
        
        # print(self.device, self.dtype, torch.isnan(sample).any(), torch.isinf(sample).any())
        sample = sample.to(self.device, self.dtype)
        target = target.to(self.device, self.dtype)

        sample_list.append(sample)
        target_list.append(target)
      
    #create batch
    sample_batch = torch.nn.utils.rnn.pad_sequence(sample_list, batch_first=True, padding_value=self.padding_value)
    target_batch = torch.nn.utils.rnn.pad_sequence(target_list, batch_first=True, padding_value=self.padding_value)
    
    return sample_batch, target_batch