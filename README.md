Breast Cancer Classification
This is a simple deep learning project for breast cancer classification. We use PyTorch, one of the leading deep learning libraries to implement and train our model. The dataset is the Breast Cancer Wisconsin (Diagnostic) dataset available on the UCI Machine Learning Repository.

Getting Started
The project is implemented in Python. The main dependencies are PyTorch for building and training the model, sklearn for splitting the dataset and generating classification reports, pandas for data manipulation, and matplotlib for plotting the training and testing losses.

The Model
The deep learning model is a multilayer perceptron with three hidden layers. The architecture is as follows:

Input layer (number of neurons equals the number of features in the dataset)
Hidden layer 1 (45 neurons, followed by Batch Normalization and ReLU activation)
Hidden layer 2 (35 neurons, followed by Batch Normalization and ReLU activation)
Hidden layer 3 (20 neurons, followed by Batch Normalization and ReLU activation)
Output layer (1 neuron)
We use Binary Cross Entropy with Logits Loss as the loss function, and Adam as the optimizer.

Training Process
The model is trained for 1000 epochs. At every tenth epoch, the model switches to evaluation mode and the performance is tested on the test dataset. The train and test losses are recorded for each epoch and can be visualized using a matplotlib plot.

In the evaluation stage, a classification report is generated using sklearn's classification_report function, providing key classification metrics.

The model weights corresponding to the best accuracy are saved for future use.

Output
The script prints out the train and test losses for each epoch and the final best classification report. It also produces a plot showing the train and test losses over epochs.

Please refer to the Python script breast_cancer_classification.py for the actual code and further comments.
