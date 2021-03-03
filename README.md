<!-- Masked Face Emotion Detection -->
<!-- CS 541: Artificial Intelligence - Final Group Project -->
<h2 align="center">Masked Face Emotion Detection</h2>
<h3 align="center">CS 541: Artificial Intelligence - Final Group Project</h3>
  
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project is to determine the degree to which facial expressions are accurately detectible while wearing a face mask.
We will be comparing the accuracy of a machine learning model trained on a dataset of unmasked faces to the same model trained on masked faces to determine the degree of inaccuracy introduced by masking of faces. The hope is to draw conclusions about the amount of nonverbal communication about our emotions is conveyed in the lower portion of the face and, thus, obscured by the face masks.



### Built With

* Python
* []()
* []()

### Data Files Attribution and Pre-processing
* Original labeled images were downloaded from kaggle: https://www.kaggle.com/msambare/fer2013
  * These images have been labeled with emotions and there are training and test sets
* Each file was then processed using the Mask the Face tool: https://github.com/aqeelanwar/MaskTheFace
  * Installation instructions on the provided GitHub repo were very helpful, just make sure you have run pip install cmake in your python virtual environment, it is a dependency that is not explicitly called out in the requirements 
* We used a very basic shell script for iterating through all the files. This resulted in having all the files that could be masked in directories where the original un-masked image is present as well as the updated masked image.
  * Note that not all images could be masked using the tool, roughly 2/3 to 1/2 could be masked and that is what we used for our training and test sets


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Necessary software downloads
* Python
* Tensorflow
  ```sh
  ...
  ```
* Etc.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/VinodhKotipalli/masked_face_emotion_detection.git
   ```
2. Install required packages
  * Install poetry for python virtual environment (skip if already installed)
    ```sh
    curl -SSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
    ```
  * configure poetry to create virtual environment in project path
    ```sh
    poetry config virtualenvs.in-project true
    ```
  * Install dependencies and create python virtual environment(.venv will have virtual python environment)
    ```sh
    cd maskedFaceEmotionDetection
    poetry install
    ```
  * To use the already created python virtual environment 
    ```sh
    poetry env use python
    ```
  
  

  
