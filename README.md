# EEG-Saliency-NN-Analysis-NMCL-2
This repository contains the implementation for a performance and saliency analysis of neural networks for multi-class mental workload classification. Note: The report is still work-in-progress, and will be included after its completion.

Here's a brief introduction of this project: We will be working with raw EEG data from MATB subtask of the COG-BCI database. Data processing, such as filtering and referencing, along with application of functional connectivity methods, namely magnitude-square coherence, will allow us to convert raw EEG data into a coherence matrices, which is a form that our machine learning models can process. Finally, after evaluation of MLP and CNN models for multi-class mental workload classification, we will produce saliency maps with the goal of deriving meaningful insights into the relationships between different regions of the brain.
## Table of Contents
- [Installation and Usage](#Installation-and-Usage)
- [Methodology](#Methodology)
- [Results and Discussion](#Results-and-Discussion)
- [Credits and Acknowledgements](#Credits-and-Acknowledgements)

## Installation and Usage
To run the programs in the repository is simple. Simply clone the repository into your local directory and you can run the python files located in the `./src` or `./tools` repository. 

### `./src` directory:
| File name | Description |
| --- | --- |
| `classification-2` | Performs binary classification of high or low mental workload on EEG features extracted from `feature_extraction.py` with `scikit-learn` models |
| `classification-3` | Performs ternary classification of high, medium, and low mental workload on EEG features extracted from `feature_extraction.py` with `scikit-learn` models |
|`feature-extraction.py` | Reads in preprocessed EEG data from `preprocessing.py` and extracts features to be used in classification |
|`models.py` | Used to define models used in classification |
| `nn-2.py` | Performs binary classification of high or low mental workload on EEG features extracted from `feature-extraction.py` with `tensorflow/keras` models |
| `nn-3.py` | Performs ternary classification of high, medium, and low mental workload on EEG features extracted from `feature-extraction.py` with `tensorflow/keras` models |
| `preprocessing.py` | Cleans raw EEG data by filtering and re-referencing the data |
| `saliency-2.py` | Produces saliency maps based on binary classification `tensorflow/keras` models |
| `saliency-3.py` | Produces saliency maps based on ternary classification `tensorflow/keras` models |

### `./tools` directory:
| File name | Description |
| --- | --- |
| `data-convert.py` | Converts the format of the raw EEG data into CSV, without preprocessing |
| `feature-extract-all.py` | A script to concurrently run `feature-extraction.py` on multiple subjects at a time |
| `plot-matrices.py` | Plots the average coherence matrices for all subjects |
| `plot-preprocessed.py` | Plots the EEG data after preprocessing |
| `plot-raw.py` | Plots the raw EEG data |
| `plot-results.py` | Plots the results/performance of different classifiers in mental workload classification |
| `test.py` | Program to test whether `tensorflow/keras` is utilizing the GPU |

## Methodology
The methodology involves EEG data collection and preprocessing, FC-based feature extraction, mental workload
prediction (in binary & ternary scenarios), and saliency map analysis. Refer to Fig. 1 for an illustration of the process. 
![image](https://github.com/Mimsqueeze/EEG-Saliency-NN-Analysis-NMCL-2/assets/101283845/68fecc38-8bd3-44ae-94e6-add6f92d2ac7)

## Results and Discussion
We found that both MLPs and CNNs performed well in classification of EEG data, especially in comparison with the performance of SVMs from prior research (which have scored ~98.3% accuracy for binary classification and ~88.7% accuracy for ternary classification). Interestingly, we found that in both binary and ternary classification of mental workload, our best CNNs marginally outperformed our best MLPs. Our best CNNs scored 98.49% and 93.92% accuracies, and our best MLPs scored 98.35% and 92.14% accuracies for binary and ternary classification respectively.

In this work, we also generated saliency maps for both binary and ternary classification, allowing us to make both general and specific observations about the relationship between different regions of the brain with each other (associated with the sensor pairs) in the context of mental workload. We can observe that between low and high mental workload, a decrease in coherence for the high-alpha band between select sensors/regions of the brain is highly important for classification. We can also observe that across low and high mental workload levels, increases in coherence for higher frequency bands between select sensors are important for classification. We will allow those more versed in the EEG field draw more concrete conclusions based on these results. Refer to the `./plots` directory for all of the plots we generated.

## Credits and Acknowledgements
Credits to Marcel F. Hinss, Emilie S. Jahanpour, Bertille Somon, Lou Pluchon, Frédéric Dehais, Raphaëlle N Roy for the [COG-BCI Dataset](https://www.nature.com/articles/s41597-022-01898-y). The COG-BCI database includes EEG recordings of 29 participants over 3 separate sessions with 4 different tasks.

Special thanks to everyone in the [The Neuromotor Control and Learning (NMCL) Laboratory](https://sph.umd.edu/research-impact/laboratories-projects-and-programs/neuromotor-control-and-learning-laboratory) for their invaluable guidance and encouragement. 
