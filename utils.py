from torch.utils.data import Dataset, DataLoader
import sklearn.datasets
import numpy as np
import pandas as pd
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
  def __init__(self, dataset):
    ## ToDo : Loads dataset for data in Filesystem
    self.images=dataset["images"]
    self.labels=dataset["labels"]

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    img = torch.tensor(self.images[index], dtype=torch.float32)
    lab = torch.tensor(self.labels[index])
    return img, lab


def load_dataset(path):
  dataset=pd.read_csv(path)
  pixelset=dataset.iloc[:, 1:].to_numpy()
  labels=dataset.label.tolist()
  images=[]
  for i in range(0, pixelset.shape[0]):
      img=pixelset[i]
      img= img.reshape(28,28)
      images.append(img)
  return images, labels



## Function to get dataloader
def get_generator(dataset = "./Data/digiData/train.csv", batch_size = 1, batch_shuffle = True, num_workers = 1, train_size=512, test_size=50):
    '''
        Discription :
            Fuction to return pytorch DataLoaders for Test, Train set.
        Returns :
            Pytorch TrainSet Dataloader, TestSet Dataloader
        ARGS :
            dataset : Specify the location of dataloder by default it takes sklean digits dataset.
            batch_size : size of batch which Dataloader will return on one iteration
            batch_shuffle : Shuffling within a batch
            num_workers : Count of CPU thread working
            train_size : size of train set
            test_size : size of test set
    '''
    param = {
    'batch_size': batch_size,
    'shuffle': False,
    'num_workers': num_workers
    }

    images, labels = load_dataset(dataset)
    train_size*=batch_size
    test_size*=batch_size
    trainSet={ 
                "images" : images[:train_size],
                "labels" : labels[:train_size]
    }
    testSet={
                "images" : images[train_size:train_size+test_size],
                "labels" : labels[train_size:train_size+test_size]
    }

    return DataLoader(DataTupple(trainSet), **param), DataLoader(DataTupple(testSet),  **param)
