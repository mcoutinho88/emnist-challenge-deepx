# EMNIST Dataset Character Recognition Challenge from DeepX

## Description

Your company is developing a software to digitize handwritten text. Your team has already developed code to extract the image of each one of the characters in a given text. You are given the task of developing a machine learning model capable of reliably translating those images into digital characters. After some research, you find the EMNIST dataset, which seems perfect for the task.

## Dataset (pulled from [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset))

The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19  and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset.

EMNIST Balanced:  131,600 characters. 47 balanced classes.

## Solution

For the solution, the following was used:

- Convolutional Neural Networks
   - 2 blocks of 2 convolutional layers followed by a 2x2 max-pooling 
- Tensorflow and Keras
- Python

## Results

This method achieved an accuracy of 88.71% on the EMNIST Balanced Dataset

---

Made by Matheus Coutinho Cavalcante
