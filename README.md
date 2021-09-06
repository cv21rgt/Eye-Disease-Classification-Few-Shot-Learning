# Introduction
Current research at the intersection of [Artificial Intelligence (AI)](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence) and eye disease diagnosis utilizes medical images from [Optical Coherence Tomography (OCT)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1531864/), [Fundus Photography (FP)](https://www.sciencedirect.com/topics/medicine-and-dentistry/fundus-photography) and [slit-lamps](https://www.aao.org/eye-health/treatments/what-is-slit-lamp) as input into [Deep Learning (DL)](https://machinelearningmastery.com/what-is-deep-learning/) models. In cases were such models make it into production in the form of a software application, their use is limited to mainly [ophthalmologists](https://www.rcophth.ac.uk/about/what-is-ophthalmology/what-is-an-ophthalmologist/) and other medical professionals who have access to medical imaging equipment that can take such images. This does not take into account regions of the world where such equipment is either scarce or not available.

To tackle this problem, this repository diverges from current research and investigates the potential use of non-medical images such as those taken by smartphones as input into a DL model capable of eye disease diagnosis. Smartphones are an abundant resource - no matter how remote an area is, you are more likely to find a smartphone capable of taking basic images than finding scientific equipment for taking OCT or FP images or even an ophthalmologists. Due to the lack of abundant annotated data, the use of non-medical images in eye disease classification is formulated as a [Few-Shot-Learning (FSL)](https://neptune.ai/blog/understanding-few-shot-learning-in-computer-vision) classification problem and solved using [Prototypical Networks](https://towardsdatascience.com/few-shot-learning-with-prototypical-networks-87949de03ccd).

## Dataset
At the time of carrying out this research the author did not have any access to a publicly available dataset of disease infected eye images taken by smartphones. As a result, the author created a new dataset by leveraging the power of Google Images. Annotated images were downloaded from online medical journals, blogs as well as websites for medical practitioners. The author also took into account issues of gender, race, geographical location as well as whether it's the right or left eye. This was done in an effort to gather a representative dataset. When it comes to FSL, training, validation and testing datasets are split using the classes. The distribution of the image data collected in this study is summarized as below. Each class contains 30 images.

Training Classes | Cataract, Glaucoma, Healthy, Pterygium, Keratoconus
---------------- |
Validation Classes | Uveitis, Stye
------------------ |
Testing Classes | Conjunctivis, Trachoma, Strabismus

The data is made available in this repo as a compressed file data.zip.

## Methodology

## Results