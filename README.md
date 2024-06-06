# P4_HilalIşık_1904010026
### Project 4 for Artificial Intelligence Applications Course

# Installation and Setup Guide

## Requirements
- **Python**: Version 3.7.6
- **CUDA-supported NVIDIA GPU**: Recommended for faster training and testing. If unavailable, the process can be performed on a CPU, though it will take longer.

## Installation Process

### Option 1: Using Visual Studio 2022
If you have Visual Studio 2022 installed, you can:
* Automatically create a virtual environment
* Install necessary packages
* Run the application from the `main.py` file in the project folder

If Visual Studio 2022 is not available, follow the steps in the "Manual Setup" section below.

### Option 2: Manual Setup

#### Dataset Installation
1. **Download the Dataset**
   - Access the dataset via this [link](https://drive.google.com/drive/folders/1WE0JB1N0teZHPjep_ibvpQuOEjlijeUs?usp=sharing).
2. **Create a Dataset Folder**
   ```sh
   mkdir dataset
   
 3. Then, download the 'Fit3D Video Dataset.rar' file from the link below and move it into this folder.
 4. Lastly, extract the file to the current location..


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

## Model and Dataset Information

## Model
  - Model Type: LSTM
  - Model Video Sequence Count: 75 Frames per Video
  - Trained Model Video Resolution: 128 x 128
## Dataset
  - Dataset: Fit3D Video Dataset
  - Number of Exercises (Classes): 47
  - Number of Videos each Class Containing: 35
  - Number of Total Videos: 1645
