from torch.utils.data import Dataset, DataLoader
import sklearn.datasets
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import torch



## For Checked black and white patter of 4 boxes
# Function to combine white and black images
def getImgBlackWhiteCross(img1, img2, img3, img4, square_size = 100):
  '''
        Discription : 
            Function produces a numpy (n, n , 3) array of square image which consists of four equal
            squares each of one will have color either white or black as specified in arguments
            
        Returned image will in the format:
            |box1   box2|
            |box3   box4| 
        ARGS :-
                img1, img2, img3, img4 : strings specifing respective color of respective box i.e.
                                            "b" for black else white
  '''
  assert img1 == "b" or img1 == "w" or img2 == "b" or img2 == "w" or img3== "b" or img3 == "w" or img4 == "b" or img4 == "w"

  if(img1 == "b"):
    img1 = np.zeros((square_size,square_size,3))
  else:
    img1 = np.ones((square_size,square_size,3))*255
  if(img2 == "b"):
    img2 = np.zeros((square_size,square_size,3))
  else:
    img2 = np.ones((square_size,square_size,3))*255
  if(img3 == "b"):
    img3 = np.zeros((square_size,square_size,3))
  else:
    img3 = np.ones((square_size,square_size,3))*255
  if(img4 == "b"):
    img4 = np.zeros((square_size,square_size,3))
  else:
    img4 = np.ones((square_size,square_size,3))*255
  
  coloredImgRow1 = np.concatenate((img1, img2),axis=1)
  coloredImgRow2 = np.concatenate((img3, img4),axis=1)
  return np.concatenate((coloredImgRow1,coloredImgRow2),axis=0)



## DataLoader class definition of pytorch
class DataTupple(Dataset):
  def __init__(self, dataset = "digit_dataset"):
    ## ToDo : Loads dataset for data in Filesystem
    if(dataset == "digit_dataset"):
      num_dataset = sklearn.datasets.load_digits()
    self.images = num_dataset.images
    self.labels = num_dataset.target

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    img = torch.tensor(self.images[index], dtype=torch.float32)
    lab = torch.tensor(self.labels[index])
    return img, lab

## Function to get dataloader
def get_generator(dataset = "digit_dataset", batch_size = 1, batch_shuffle = True, num_workers = 1):
    '''
        Discription :
            Fuction to return pytorch DataLoader for Dataset.
        Returns :
            Pytorch Dataloader,
        ARGS :
            dataset : Specify the location of dataloder by default it takes sklean digits dataset.
            batch_size : size of batch which Dataloader will return on one iteration
            batch_shuffle : Shuffling within a batch
            num_workers : Count of CPU thread working
    '''
    param = {
    'batch_size': batch_size,
    'shuffle': True,
    'num_workers': num_workers
    }
    loader = DataTupple(dataset)
    return DataLoader(loader, **param)
