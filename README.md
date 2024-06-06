# GymAIner
### Gym AI Trainer for Artificial Intelligence Applications Course

## Requirements
  - Python 3.7.6
  - Cuda supported NVIDIA GPU if possible for faster training and testing processes.
    - Doing this process on CPU instead will result only in longer time.


# Installation Process
## Installation via Visual Studio 2022
  - If you have Visual Studio 2022 installed, you can;
	- Automatically create virtual environment,
	- Install packages and,
	- Run the application from the main.py file in the project folder.

  - If not, please follow the steps in "Dataset Installation" sections below.


## Dataset Installation
  ### Dataset Link: https://drive.google.com/drive/folders/1WE0JB1N0teZHPjep_ibvpQuOEjlijeUs?usp=sharing
  - Please create a folder named 'dataset' in the project directory.
  - Then, download the 'Fit3D Video Dataset.rar' file from the link below and move it into this folder.
  - Lastly, extract the file to the current location..")


## Required Libraries
  - Please install the required packages by running the following command in the terminal:
    - Create a virtual environment:
	```py -3.7 -m venv venv```

	- Activate the virtual environment:
	```.\venv\Scripts\activate``` for Windows
	```source venv/bin/activate``` for MacOS and Linux

	- Install the required packages:
	```pip install -r requirements.txt```


## Run Application
  - Run the following command after package installations are completed:
  ```python main.py```
  - After running, navigate through the console for desired operations.


# Project Details
## Dataset
  - Dataset: Fit3D Video Dataset
  - Number of Exercises (Classes): 47
  - Number of Videos each Class Containing: 35
  - Number of Total Videos: 1645

## Model
  - Model Type: LSTM
  - Model Video Sequence Count: 75 Frames per Video
  - Trained Model Video Resolution: 128 x 128
