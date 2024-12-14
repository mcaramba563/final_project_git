# Perceptron Neural Network Application

This repository contains a neural network application implementing a multilayer perceptron for image recognition tasks, such as digit recognition using the MNIST dataset. The app allows you to train, predict, save, and load models easily.

---

## Features

- **Create Custom Models**: Define the number of neurons in hidden layers.
- **Train on Custom Data**: Train the model using your own dataset.
- **Train on Random Images**: Train the model on randomly selected images.
- **Make Predictions**: Predict the class label for an input image.
- **Save and Load Models**: Save the current model to disk and load it for later use.
- **Load Default Model**: Reset the model to the default pretrained configuration.
- **Reset Training**: Revert the current model to its state before any training.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mcaramba563/final_project_git.git
    cd final_project_git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

# Usage

## Start the App
Run the app in interactive mode:
    
    python main.py

# Features and Commands
## 1. Create a Custom Model
Create a model with specific hidden layer sizes:

    make_custom_model layer1_size layer2_size ...

Example:

    make_custom_model 400 256 128

## 2. Train on Custom Data
Train the model on labeled data from a specified file(the file containes pathes to images and labels):

    train path_to_dataset_file number_of_epochs learning_rate


Example:

    train dataset/train_data.txt 5 0.01
where dataset/train_data.txt:

    images/mnist_png/test/0/10.png 0
## 3. Train on Random Images from default dataset
Train the model on randomly selected images:

    train_on_random_images number_of_images number_of_epochs learning_rate

Example:

    train_on_random_images 20000 2 0.01

## 4. Make Predictions
Predict the class of a given image:

    predict path_to_image

Example:

    predict images/mnist_png/test/0/10.png

## 5. Save the Model
Save the current model to a file:

    save_model path_to_model_file

Example:

    save_model models/my_model.npy

## 6. Load a Custom Model
Load a previously saved model:

    load_custom_model path_to_model_file

Example:

    load_custom_model models/my_model.npy

## 7. Load Default Model
Reset the model to its default pretrained state:

    load_default_model

## 8. Reset Training
Revert the current model to its initial state before training:

    reset_training

## 9. Exit the Application
Exit the interactive mode:

    exit

# Example Workflow
1. Start the app:

    ```python main.py```

2. Create a custom model with hidden layers of sizes 400, 256, and 128:

    ```make_custom_model 400 256 128```

3. Train on custom data for 5 epochs with a learning rate of 0.01:

    ```train dataset/train_data.txt 5 0.01```

4. Save the trained model:

    ```save_model models/my_trained_model.npy```

5. Load the default pretrained model:

    ```load_default_model```

6. Predict the class of a test image:

    ```predict images/mnist_png/test/0/10.png```

7. Exit the app:

    ```exit```