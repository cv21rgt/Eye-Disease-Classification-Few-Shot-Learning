import torch

# paths to the training, validation and test data sets
train_data_dir = "/home/rodney/Documents/PyTorch/Classification/Projects/Eyes_Diseases/Data_For_Few_Shot_Learning/training_data"
validation_data_dir = "/home/rodney/Documents/PyTorch/Classification/Projects/Eyes_Diseases/Data_For_Few_Shot_Learning/validation_data"
test_data_dir = "/home/rodney/Documents/PyTorch/Classification/Projects/Eyes_Diseases/Data_For_Few_Shot_Learning/test_data"

# create tensors for mean and standard deviation of the image data
# these will be used to standardize the data
MEAN = torch.tensor([0.5507, 0.4053, 0.3529]) # these were calculated for this dataset
STD = torch.tensor([0.2550, 0.2261, 0.2246])