import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Function to compute the Euclidean distance between feature vectors
def euclidean_dist(x, y):
  """
  Computes euclidean distance btw x and y
  Args:
      x (torch.Tensor): shape (n, d). n usually n_way*n_query
      y (torch.Tensor): shape (m, d). m usually n_way
  Returns:
      torch.Tensor: shape(n, m). For each query, the distances to each centroid
  """
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)


class ProtoNet(nn.Module):
  """
  This class implements a Prototypical Network. It is made up of three parts.

    1. Feature extractor part - this uses a CNN e.g ResNet, VGG, DenseNet or a user supplied smaller network to extract 
       the most important features of each image and convert them into a vector/embedding.
    2. For each class in the support set, the vectors of the images are averaged to compute the class prototype, that 
       represents that class on a feature space.
    3. For each image in the query set, a vector is extracted as in (1) and then the Euclidean distance is computed 
       between the query image vector and the class prototypes. The shorter the distance, the more likely that the 
       query image belongs to that class. 
       
  """
  def __init__(self, encoder, device="cpu"):
    """
    Args:
        encoder : CNN that extracts the image features and turns them into a vector/embedding
        n_way (int): number of classes in a classification task
        n_support (int): number of images per class in the support set
        n_query (int): number of images per class in the query set
    """
    super(ProtoNet, self).__init__()

    self.device = device
    self.encoder = encoder.to(self.device)

  def set_forward_loss(self, sample):
    """
    Computes loss, accuracy and output for classification task
    Args:
        sample (torch.Tensor): shape (n_way, n_support+n_query, (dim=[C, H, W])) 
    Returns:
        torch.Tensor: shape(2), loss, accuracy and y_hat
    """
    sample_images = sample['images'].to(self.device) # retrieve the support + query images
    n_way = sample['n_way']                 # get no. of classes in sample
    n_support = sample['n_support']         # no. of support images in each class     
    n_query = sample['n_query']             # no. of query images in each class

    # seperate the support and query images
    x_support = sample_images[:, :n_support] 
    x_query = sample_images[:, n_support:]
   
    #target indices are 0 ... n_way-1
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long() # result = torch.Size([n_way, n_query, 1])
    target_inds = Variable(target_inds, requires_grad=False) 
    target_inds = target_inds.to(self.device)
   
    #encode images of the support and the query set
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)
    
    z = self.encoder.forward(x) # returns embedded vector
    z_dim = z.size(-1) #get the size of the flattenned vector
    # find the mean, that becomes the class prototype
    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)    
    z_query = z[n_way*n_support:]

    #compute distances between vectors of images in the query set and the class prototypes
    dists = euclidean_dist(z_query, z_proto)
    
    #compute probabilities
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
   
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
   
    return loss_val, {
        'loss': loss_val.item(),
        'acc': acc_val.item(),
        'y_hat': y_hat
        }