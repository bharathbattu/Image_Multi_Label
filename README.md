ImageMultiLabel
A multiclass image classification project using a Convolutional Neural Network (CNN) built with TensorFlow to classify images of flowers into four categories.
Project Overview
This project implements a CNN model for multiclass image classification, trained on a dataset of flower images. The model uses TensorFlow and Keras to preprocess images, build a sequential CNN architecture, and train it to classify images into one of four classes. The project is implemented in a Jupyter Notebook (ImageMultiLabel.ipynb).
Features

Dataset: Flower images organized in a directory with four classes.
Preprocessing: Image augmentation (shear, zoom, horizontal flip) and rescaling for robust training.
Model: A CNN with three convolutional layers, max-pooling layers, and a dense output layer with softmax activation.
Training: Trained for 20 epochs with Adam optimizer and categorical cross-entropy loss.
Evaluation: Validation accuracy of approximately 71.74% after 20 epochs.

Requirements
To run this project, you need the following dependencies:

Python 3.7+
TensorFlow 2.x
NumPy
Jupyter Notebook

Install the required packages using pip:
pip install tensorflow numpy jupyter

Dataset
The dataset consists of 431 training images and 46 validation images, divided into 4 classes, stored in the directory C:\Users\Asus\flower. Ensure your dataset is organized in subdirectories, with each subdirectory representing a class.
Example directory structure:
flower/
├── class1/
├── class2/
├── class3/
├── class4/

Update the base_dir variable in the notebook to point to your dataset directory.
Model Architecture
The CNN model is built using TensorFlow's Keras API with the following layers:

Conv2D: 64 filters, 3x3 kernel, stride=2, ReLU activation, same padding
MaxPool2D: 2x2 pool size, stride=2
Conv2D: 32 filters, 3x3 kernel, stride=2, ReLU activation, same padding
MaxPool2D: 2x2 pool size, stride=2
Conv2D: 32 filters, 3x3 kernel, stride=2, ReLU activation, same padding
MaxPool2D: 2x2 pool size
Flatten: Converts 2D feature maps to 1D
Dense: 4 units with softmax activation for 4-class classification

Training
The model is trained for 20 epochs with the following configuration:

Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Batch Size: 64
Image Size: 224x224 pixels
Data Augmentation: Shear (0.2), zoom (0.2), horizontal flip
Validation Split: 10% of the dataset

Training results show a final validation accuracy of ~71.74% and a validation loss of ~0.7286.
Usage

Clone the repository:
git clone https://github.com/your-username/ImageMultiLabel.git
cd ImageMultiLabel


Install dependencies:
pip install -r requirements.txt


Update the base_dir in ImageMultiLabel.ipynb to point to your dataset directory.

Run the Jupyter Notebook:
jupyter notebook ImageMultiLabel.ipynb


Execute the cells to preprocess the data, build, and train the model.


Results
After 20 epochs, the model achieves:

Training accuracy: ~68.83%
Validation accuracy: ~71.74%
Validation loss: ~0.7286

Future Improvements

Experiment with deeper architectures or transfer learning (e.g., using pre-trained models like VGG16 or ResNet).
Increase dataset size or apply more advanced augmentation techniques.
Tune hyperparameters (e.g., learning rate, batch size, number of epochs).
Add dropout layers to prevent overfitting.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.
License
This project is licensed under the MIT License.
