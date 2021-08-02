import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset


def check_image_path(directory_with_images):
  """
  Checks for broken file paths to images and removes them from our final dataset

    Args:
      directory_with_images (str): path to directory that holds the image data

    Returns:
      correct_filepaths (list): a list of unbroken file paths to images
  """

  correct_filepaths = []

  # get a list of image names in directory
  list_of_images = os.listdir(directory_with_images) 
  for img in list_of_images:
    img_filePath = directory_with_images + img
    # check if we can successfully open the image
    if cv2.imread(img_filePath) is not None:
      # only append file paths to images we can successfully open
      correct_filepaths.append(img_filePath)

  return correct_filepaths


def read_eye_images(directory_path, img_height, img_width, augmentation=False):
  """
  Reads image data from directories, resizes images to same size & performs augmentation to increase data size
  in terms of classes.

    Args:
      directory_path (str): path to directory that holds the image data
      img_height (int): height of image to resize to
      img_width (int): width of image to resize to
      augmentation (bool): whether to perform class augmentation or not

    Returns:
      datax (np.array): NumPy array of images
      datay (np.array): NumPy array of classes/labels
  """

  datax = [] # list to hold the images
  datay = [] # list to hold the image labels

  # get a list of the sub_directories in the directory_path
  sub_directories = os.listdir(directory_path)

  for sub_dir in sub_directories: # for each sub_directory read the images

    # check that image file paths are valid, we do not want to deal with broken paths
    correct_image_filepaths = check_image_path(directory_path + '/' + sub_dir + '/')
    
    for fpath in correct_image_filepaths: # loop through the image file paths

      # read the image 
      image = cv2.imread(fpath) # this reads image in BGR format

      # convert to RGB
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # resize image to a uniform size
      image = cv2.resize(image, (img_width, img_height))
      
      # perform data augmentation in the form of rotations and flips to increase the classes in our train data dataset
      if augmentation:
        rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
        rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        flip_vertical = cv2.flip(image, 0)
        flip_horizontal = cv2.flip(image, 1)
        flip_both = cv2.flip(image, -1)      
            
        # add image data to our 'datax' list. Using 'extend' concatenates the data. Use it with another iterable 
        # e.g a list
        datax.extend((image, rotated_90, rotated_180, rotated_270, flip_vertical, flip_horizontal, flip_both))
        # add the labels to our list
        datay.extend((
          sub_dir + '_0', 
          sub_dir + '_90', 
          sub_dir + '_180', 
          sub_dir + '_270',
          sub_dir + '_FV',
          sub_dir + '_FH',
          sub_dir + '_fVH'          
          ))

      else: # if its any other data e.g. validation, test data, we are not augmenting it
        datax.append(image)
        datay.append(sub_dir)     

  return np.array(datax), np.array(datay)

def display_images(images, labels, rows = 1, cols=1):
    """
    Display a single or multiple images.

        Args:
            images (list): list of images to display
            labels (list): list of labels/classes for the images
            rows (int): how many rows of images to display
            cols (int): how many columns of images to display
    """

    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for i, (img, label) in enumerate(zip(images, labels)):
        ax.ravel()[i].imshow(img)
        ax.ravel()[i].set_title(label)
        ax.ravel()[i].set_axis_off()
    
    plt.show()

# function to compute the mean and standard deviation of our data
def get_mean_std(image_data_directory, img_height, img_width, batch_size=1):
  """
  Compute the mean and standard deviation of an image dataset.
  
    Args:
      image_data_directory (str): path to directory containing the image data
      img_height (int): height to resize all images to
      img_width (int): width to resize all images to 
      batch_size (int): how many images to process at a time especially if dealing with a large dataset

    Returns:
      mean (float): mean of the dataset
      std (float): standard deviation of the dataset
  """

  # prepare the training data
  unbroken_images_filepaths = check_image_path(image_data_directory) 

  print(f"Number of images used to compute the mean and standard deviation: {len(unbroken_images_filepaths)}")  
  
  # create an object of type DataPreProcessing
  image_data = DataPreProcessing(unbroken_images_filepaths, img_height, img_width)  
  
  
  # invoke a dataloader that returns data points == batch_size at random
  # You can also use drop_last=True, if your data size is not divisible by the batch size. 
  # This means the last batch will be smaller and will not be used.
  image_dl = DataLoader(image_data, batch_size=batch_size, shuffle=False, drop_last=True) 
  

  channels_sum, channels_squared_sum, num_batches = 0, 0, 0

  for images in image_dl:

    channels_sum += torch.mean(images * 1.0, dim=[0, 2, 3])

    channels_squared_sum += torch.mean((images * 1.0)**2, dim=[0, 2, 3])

    num_batches += 1

  mean = channels_sum / num_batches

  std = (channels_squared_sum/num_batches - mean**2)**0.5

  return mean, std

class DataPreProcessing(Dataset):
  """
  This class takes images, resizes each image to the same size, converts each image to type float32, 
  scales the pixel values to [0,1] and converts each image to a Tensor that can be input into a PyTorch model 
  """
  def __init__(self, images_filepaths, img_height, img_width):
    """
      Args:
        images_filepaths (list): list of unbroken image filepaths. You can use the check_image_path()
                                 to obtain such a list
        img_height (int): height to resize all images to
        img_width (int): width to resize all images to
      Returns:
        image (tensor): image tensor of dimension [channels x height x width]  
    """
    
    self.images_filepaths = images_filepaths

    self.height = img_height

    self.width = img_width
    
  
  def __len__(self):
    """
    Returns the number of images in dataset
    """
    return len(self.images_filepaths)


  def __getitem__(self, ix):
    """
    Gives you access to an individual image through the index (ix)
    """

    # Pre-processing the image
    # ------------------------
    image_filepath = self.images_filepaths[ix] # get path to image

    image = cv2.imread(image_filepath) # image read in BGR format

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image RGB format

    image = cv2.resize(image, (self.height, self.width)) # resize images to the same size

    image = torch.from_numpy(image).float() # convert to Tensor and float data

    image = image / 255 # scale image to [0, 1]

    # change dimensions from [H, W, C] to [C, H, W], the expected format for PyTorch image data
    image = image.permute(2, 0, 1) 
          
    return image


