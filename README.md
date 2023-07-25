# Unleashing the Power of Deep Reinforcement Learning for Early Stress Detection in Multimodal Time Series Data
This repository contains the implementation of early stress detection using Deep Reinforcement Learning (DRL). The proposed method utilizes the power of DRL to dynamically analyze small segments of time series data, enabling efficient and real-time stress classification.

The key objective is to develop an adaptive stress detection system with high accuracy and minimal classification time. We achieve this through a dynamic observation window strategy, allowing the DRL agent to adjust its observation approach based on time series complexity. The approach is comprehensively evaluated using a Leave-One-Subject-Out (LOSO) validation approach.


## Early Stress Detection Features

- Utilizes the power of Deep Reinforcement Learning (DRL) for early stress detection.
- Dynamic observation window strategy for efficient and real-time stress classification.
- Adaptive stress detection system achieving high accuracy.
- Extensive validation in LOSO

## Installation

### Clone the repository

```bash
git clone <repository_url>
cd <repository_name>

```

### Install dependencies

#### Using virtualenv
1) First create a Python virtual environment (optional) using the following command:
```bash
python -m venv stress_classification_env
```
2) Activate the virtual environment using the following command:
```bash
source stress_classification_env/bin/activate
```
3) Install the dependencies using the following command:
```bash
pip install -r requirements.txt
```
#### Using Conda
For Conda users, you can create a new Conda environment using the following command:
```bash
conda env create -f environment.yml
```
Then, activate the environment using the following command:
```bash
conda activate stress_classification_env
```

The code was tested with Python 3.8.6, PyTorch 1.9.1, CUDA 11.1 and Ubuntu 22.04.2 LTS.
For more informations about the requirements, please check the requirements.txt.
All the experiments used a single NVIDIA A100 GPU. It may take several days of training to reproduce the whole experiment in leave-one-subject-out (LOSO)

## Usage
### Prerequisites

To reproduce the entire experiment, simply run the main.py file. You can customize the experiment settings by editing the following parameters in the main.py file:

- PATH: Set the desired location where the results will be saved. By default, the results will be stored in the Results folder for each subject in the dataset.

- NUM_CLASSES: Specify the number of classes for the stress detection task. By default, the experiment is conducted with 4 classes.

## Results
Upon running the code, all the results are automatically saved in the Results folder, organized for each subject in the dataset. The Results folder contains essential outputs that facilitate the evaluation and analysis of the stress classification model.

- tensorboard log file, which captures the evolution of various training metrics throughout the learning process. Key metrics, such as episode_reward_mean and episode_len_mean, are recorded, providing insights into the model's learning progress and performance.

- A CSV file for each subject in the dataset. This file includes the model's predictions, the corresponding true labels, and the classification time for each instance. With this CSV file, users can easily compute essential evaluation metrics such as the classification report and confusion matrix for each subject.


## Citation
If you use this code in your research, please cite our paper: .....


## Contact for Issues

If you have any questions or if you are just interested in having a virtual coffee about Generative AI, 
please don't hesitate to reach out to me at: [leonardo.furia@unicampus.it]

May be the AI be with you!

## License

This code is released under the MIT License.