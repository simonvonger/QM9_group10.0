# Deep Learning Project in Molecular Property Prediction using Graph Neural Networks

## Authors
- Katrine Bay - s183910
- Simon Vonger Berg - s194232
- Hans Forum Møller - s194169


## Overview
This project has implemented graph neural network _the Polarizable Atom Interaction Neural Network_ (PaiNN) model, for predicting the chemical properties of over 130,000 organic molecules in the Quantum Machine 9 (QM9) dataset. The implementation of the model is the final project of the Deep Learning course at DTU, and the objective is to enhance the efficiency in molecular property prediction.

The use of deep learning for molecular predictions has greatly improved the nonlinear relationship between structure and properties. This project uses the PaiNN model, optimized using PyTorch and CUDA on DTU's high-performance computer (HPC) GPUs, to predict various chemical properties. The project's results indicate the PaiNN model's successful application in chemical property prediction.



## Key Features
- Implementation of the PaiNN model.
- Training and validation of four model instances on the QM9 dataset.
- Evaluation of the model's accuracy using the metric Mean Absolute Error (MAE).

## Getting Started
### Prerequisites
- PyTorch
- CUDA for GPU acceleration

### Installation
1. Clone the repository: `git clone https://github.com/simonvonger/QM9_group10.0`
2. Install the required dependencies: `pip install -r requirements.txt`



## Implementation Details
The project includes a Python script (`Main.py`) that starts the model training process using PyTorch. The PaiNN model is implemented with CUDA support for efficient GPU utilization. The script handles data loading, model training, saving the best model, and testing the trained model.


## Results
The models demonstrate effective learning with the training and validation loss curves. The achieved MAE values for properties like polarizability, ZPVE, LUMO, and HOMO underscore the model's predictive accuracy. The plots for the evaluation are displayed in the jupyter notebook (`Main_Pred.ipynb`).
