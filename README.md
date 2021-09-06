# Few-shot image classification of eye diseases using Prototypical Networks: An experimental use of non-medical images
This repository contains data, code and results from carrying out the above titled research.

## Introduction
Current research at the intersection of [Artificial Intelligence (AI)](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence) and eye disease diagnosis utilizes medical images from [Optical Coherence Tomography (OCT)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1531864/), [Fundus Photography (FP)](https://www.sciencedirect.com/topics/medicine-and-dentistry/fundus-photography) and [slit-lamps](https://www.aao.org/eye-health/treatments/what-is-slit-lamp) as input into [Deep Learning (DL)](https://machinelearningmastery.com/what-is-deep-learning/) models. In cases were such models make it into production in the form of a software application, their use is limited to mainly [ophthalmologists](https://www.rcophth.ac.uk/about/what-is-ophthalmology/what-is-an-ophthalmologist/) and other medical professionals who have access to medical imaging equipment that can take such images. This does not take into account regions of the world where such equipment is either scarce or not available.

To tackle this problem, this repository diverges from current research and investigates the potential use of non-medical images such as those taken by smartphones as input into a DL model capable of eye disease diagnosis. Smartphones are an abundant resource - no matter how remote an area is, you are more likely to find a smartphone capable of taking basic images than finding scientific equipment for taking OCT or FP images or even an ophthalmologists. Due to the lack of abundant annotated data, the use of non-medical images in eye disease classification is formulated as a [Few-Shot-Learning (FSL)](https://neptune.ai/blog/understanding-few-shot-learning-in-computer-vision) classification problem and solved using [Prototypical Networks](https://towardsdatascience.com/few-shot-learning-with-prototypical-networks-87949de03ccd).

## Dataset
At the time of carrying out this research the author did not have any access to a publicly available dataset of disease infected eye images taken by smartphones. As a result, the author created a new dataset by leveraging the power of Google Images. Annotated images were downloaded from online medical journals, blogs as well as websites for medical practitioners. The author also took into account issues of gender, race, geographical location as well as whether it's the right or left eye. This was done in an effort to gather a representative dataset. When it comes to FSL, training, validation and testing datasets are split using the classes. The distribution of the image data collected in this study is summarized as below. Each class contains 30 images.

|Data | Classes|
|-----|--------|
|Training | Cataract, Glaucoma, Healthy, Pterygium, Keratoconus|
|Validation | Uveitis, Stye|
|Testing | Conjunctivis, Trachoma, Strabismus|

The data is made available in this repo as a compressed file **data.zip**.

## Methodology
The FSL approach used in this study is based on Prototypical Networks proposed by [Snell et. al (2017)](https://arxiv.org/abs/1703.05175). The author used [transfer learning](https://www.allerin.com/blog/how-to-fine-tune-your-artificial-intelligence-algorithms) and [fine tuning](https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/). In both cases [VGG19](https://arxiv.org/abs/1409.1556), [ResNet50](https://arxiv.org/abs/1512.03385) and [DenseNet121](https://arxiv.org/abs/1608.06993) deep learning models where used as the feature extractors and the results compared. See the Jupyter notebooks **Transfer_Learning.ipynb** and **Fine_Tuning.ipynb** for code. The Euclidean distance was used as the metric for classifying query images.

## Results
*a. FSL test accuracy (%) when using transfer learning for feature extraction*

Since there is no model training during transfer learning, the Prototypical Network was simply evaluated on the eye image test dataset. For each test, the model was allowed to see 1 or 5 examples per class (Support set) and each time asked to predict or classify 10 randomly selected images (Query set). The test results are summarized as follows:

|Pre-trained network used as feature extractor| 2-way 1-shot| 2-way 5-shot| 3-way 1-shot | 3-way 5-shot |
|---------------------------------------------|-------|-------|-------|-------|
|VGG19                                        | 72.24 | 91.18 | 63.66 | 84.05 |
|ResNet50                                     | 80.11 | 92.26 | 69.08 | 86.42 |
|DenseNet121                                  | 84.89 | 95.29 | 75.58 | 91.05 |

*b. FSL test accuracy (%) when using a fine-tuned model as a feature extractor*

Using a fine-tuned model meant that part of the pre-trained models (VGG19, ResNet50 and DenseNet121) were trained on the eye image dataset as part of the Prototypical Network. By keeping track of the accuracy and loss values of the training and validation datasets, the author was able to keep track of the Prototypical Network performance during training. Unfortunately, when using the VGG19 and ResNet50 models, the network overfit the data and hence were not used on the testing data. Th following table shows the test results from a Prototypical Network using a fine-tuned DenseNet121 model as the feature extractor.

|Fine-tuned network used as feature extractor| 2-way 1-shot| 2-way 5-shot| 3-way 1-shot | 3-way 5-shot |
|---------------------------------------------|-------|-------|-------|-------|
|DenseNet121                                  | 81.06 | 93.60 | 71.34 | 88.23 |