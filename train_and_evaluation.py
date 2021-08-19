import numpy as np
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt 
from tqdm import tqdm_notebook
from tqdm import tnrange

from utilities import MEAN, STD


# create samples
def extract_sample(n_way, n_support, n_query, datax, datay):
  """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task to sample
      n_support (int): number of images per class in the support set
      n_query (int): number of images per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
      
  Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
        (NumPy array): class_labels (labels of the randomly selected classes)
  """
  sample = []

  # from Total No. of classes in an array of data (np.unique(datay)), randomly select n_way classes
  K = np.random.choice(np.unique(datay), n_way, replace=False) # returns a numpy array with a size equal to n_way

  for cls in K: # cls = data class

    datax_cls = datax[datay == cls] # get the images corresponding to our class

    perm = np.random.permutation(datax_cls) # randomly shuffle the images of this class

    sample_cls = perm[:(n_support+n_query)] #  select a sample, which is == (no. of support images + no. of query images)

    sample.append(sample_cls) # add sample images to list, we end up with [[images of cls_1],[images of cls_2],..., [images of class cls_n_way]]

  sample = np.array(sample) # convert list to numpy array

  sample = torch.from_numpy(sample) # convert to a tensor

  sample = sample.type(torch.float32) / 255.0 # convert to float and scale the image data to [0, 1]

  # The above sample has dimensions [n_way, n_support+n_query, img_height, img_width, channels]
  # re-arange the dimensions so that channels are first
  sample = sample.permute(4, 2, 3, 0, 1) # [channels, img_height, img_width, n_way, n_support+n_query]

  # standardize the data by subtracting the mean and dividing by the standard deviation
  sample = (sample - MEAN[:,None, None, None, None]) / STD[:, None, None, None, None]

  # rearange the dimensions to the original ones [n_way, n_support+n_query, img_height, img_width, channels]
  sample = sample.permute(3, 4, 1, 2, 0)

  # Since we are using PyTorch, input into our model should be of the form  [n_way, n_support+n_query, channels, img_height, img_width]
  sample = sample.permute(0,1,4,2,3)
  
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'class_labels': K
      })


def display_sample(sample):
  """
  Displays sample in a grid

  Args:
      sample (torch.Tensor): sample of images to display      
  """
  #need 4D tensor to create grid, currently 5D
  sample_4D = sample.view(sample.shape[0]*sample.shape[1],*sample.shape[2:])  

  #make a grid
  out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
  plt.figure(figsize=(16,7))

  # out has dimension [channels, img_height, img_width]

  out = out.permute(1, 2, 0) # [img_height, img_width, channels]
  out = out * STD[None, None, :] + MEAN[None, None, :] # remember you are only multiplying by the channels dimension

  plt.imshow(out)



def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, epoch, train_episode):
  """
  Trains the Prototypical Network

  Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      epoch (int): number of epochs run so far      
      train_episode (int): episodes per epoch      
  """
  # place the model in "training mode", which turns on 'dropout' & 'batch normalization' 
  model.train()

  running_loss = 0.0
  running_acc = 0.0

  # enforce regularization on the weights and bias parameters across all layers
  #L1_regularization = 0
  L2_regulization = 0

  for episode in tnrange(train_episode, desc=f"Epoch {epoch+1} train: "):

    sample = extract_sample(n_way, n_support, n_query, train_x, train_y)

    optimizer.zero_grad()

    loss, output = model.set_forward_loss(sample)

    # to reduce overfitting, we will pernalize the model for high weight values during training
    # by applying regularization
    #L1_regularization = torch.tensor(0., requires_grad=True) 
    L2_regularization = torch.tensor(0., requires_grad=True)  
    for param in model.parameters():
      #L1_regularization = L1_regularization + torch.norm(param, 1) # get the absolute value of the weight & bias values across layers
      L2_regularization = L2_regularization + torch.norm(param, 2)

    # use L1 regularization
    #loss = loss + (0.0001 * L1_regularization) 
    # use L2 regularization
    loss = loss + (0.01 * L2_regularization)
    # use both
    #loss = loss + (0.0001 * L1_regularization) +(0.01 * L2_regularization)

    running_loss += output['loss']
    running_acc += output['acc']
    
    loss.backward()
    optimizer.step()

  avg_loss = running_loss / train_episode
  avg_acc = running_acc / train_episode

  return avg_loss, avg_acc

def validate(model, validation_x, validation_y, n_way, n_support, n_query, epoch, validation_episode):
  """
  Evaluates the Prototypical Network

  Args:
      model: trained model
      validation_x (np.array): images of testing set
      validation_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      epoch (int): current epoch being run
      validation_episode (int): number of episodes to test on      
  """
  running_loss = 0.0
  running_acc = 0.0

  # place the model in "evaluation mode", which turns on 'dropout' & 'batch normalization' 
  model.eval()

  with torch.no_grad(): # we are not computing any gradients, this uses less memory and speeds up computations

    for episode in tnrange(validation_episode, desc=f"Epoch {epoch+1} validation: "):
      sample = extract_sample(n_way, n_support, n_query, validation_x, validation_y)
      loss, output = model.set_forward_loss(sample)
      running_loss += output['loss']
      running_acc += output['acc']
      
    avg_loss = running_loss / validation_episode
    avg_acc = running_acc / validation_episode
  
  return avg_loss, avg_acc

def test_model_on_one_task(model, n_way, n_support, n_query, test_episodes, x_test, y_test):
  """
  Tests the Prototypical Netweork on a test set

  Args:
      model: trained model      
      n_way (int): number of classes in a classification task
      n_support (int): number of images per class in the support set
      n_query (int): number of images per class in the query set      
      test_episodes (int): number of episodes to test on
      x_test (np.array): images of testing set
      y_test (np.array): labels of testing set      

  Returns:
    avg_loss (float): average loss
    avg_acc (float): average accuracy
  """

  running_loss = 0.0
  running_acc = 0.0

  # place the model in "evaluation mode" 
  model.eval()

  print(f"Test loss and accuracy every 100 episodes: ")
  with torch.no_grad(): # we are not computing any gradients, this uses less memory and speeds up computations

    for episode in range(test_episodes):
      sample = extract_sample(n_way, n_support, n_query, x_test, y_test)
      loss, output = model.set_forward_loss(sample)
      running_loss += output['loss']
      running_acc += output['acc']

      if (episode % 100 == 0):
        print(f"Episode: {episode} ---> Loss: {output['loss']:.3f}, Accuracy: {output['acc']:.2f}")
      
    avg_loss = running_loss / test_episodes
    avg_acc = running_acc / test_episodes
  
  return avg_loss, avg_acc


def run_training_and_evaluation(model, 
                                train_x, 
                                train_y, 
                                validation_x, 
                                validation_y, 
                                n_way, 
                                n_support, 
                                n_query, 
                                train_episode, 
                                validation_episode,
                                optimizer,
                                max_epoch,
                                filename                                
                                ):
  """
  Runs both training and evaluation per epoch

  Args:
      model: trained model
      train_x (np.array): images of training set
      train_y (np.array): labels of training set
      validation_x (np.array): images of testing set
      validation_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      training_episode (int): number of episodes to train on
      validation_episode (int): number of episodes to validate on
      optimizer : optimiser for training
      max_epoch (int): maximum number of epochs   
      filename (str): name of best model to be saved 
            
  Returns:
      train_loss_list           (list): training loss values from epoch=1 to max_epoch
      train_accuracy_list       (list): training accuracy values from epoch=1 to max_epoch
      validation_loss_list      (list): validation loss values from epoch=1 to max_epoch
      validation_accuracy_list  (list): validation accuracy values from epoch=1 to max_epoch
  """

  best_validation_loss = float('inf')

  # lists to hold the training loss & accuracy values for all epochs
  train_loss_list, train_accuracy_list = [], []
  # lists to hold the validation loss & accuracy values for all epochs
  validation_loss_list, validation_accuracy_list = [], []

  #divide the learning rate by 2 at each epoch, as suggested in paper
  scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)

  print(f"Start training: ")

  for epoch in range(max_epoch):  

    train_loss, train_accuracy = train(model, optimizer, train_x, train_y, n_way, n_support, n_query, epoch, train_episode)
    validation_loss, validation_accuracy = validate(model, validation_x, validation_y, n_way, n_support, n_query, epoch, validation_episode)

    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      torch.save(model.state_dict(), filename)

    print(f"\nEpoch: {epoch + 1}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_accuracy*100:.2f}%")
    print(f"\t Val. Loss: {validation_loss:.3f} | Val. Acc: {validation_accuracy*100:.2f}%")

    # append losses and accuracies to list
    # these will be used for plotting Loss/Acc vs Epochs
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)
    validation_loss_list.append(validation_loss)
    validation_accuracy_list.append(validation_accuracy)

    scheduler.step()

  return train_loss_list, train_accuracy_list, validation_loss_list, validation_accuracy_list 


# Prediction on a sample of data
def predict(model, sample, device="cpu"):
  """
    Args:
      model (object): trained prototypical model
      sample (dict): dictionary containing the following keys:
                                                              images - images for the support + query set
                                                              n_way - number of classes to sample
                                                              n_support - number of support images
                                                              n_query - number of query images 
      device (str): device to run the model on - 'cpu' or 'cuda'

    Returns:
      output (dict): dictionary with the following keys:
                                                        loss - loss value
                                                        acc - accuracy of prediction
                                                        y_hat = prediction tensor for each query image in each class
  """
  model.to(device)
  l, output = model.set_forward_loss(sample)

  return output