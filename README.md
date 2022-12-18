# ArcFace-Project
Project For CPS

** TO RUN ARCFACETEST.py **
arcfacetest.py requires the source code from https://github.com/serengil/deepface, to run the program place both arcfacetest.py and the images folder in the main directory

** TO RUN ArcFace_Comparison.ipynb **

** Please note that this file must be run in Google Collab instead of Jupyter Notebook. This is likely due to the version of conda that is used in Jupyter Notebook compared to Google Collab.
** The current file loads a run of OpenFace using MTCNN for non matching images. If you would like to change the images or models, please find instructions below:

1. For the image sets, please upload images found in the 'images' folder and change the names of the desired files you'd like to use in cell 3 of the notebook
2. To modify the models used, you can find the option in cell 8 with variable named "model_name"
3. The variable "resp" then stores the JSON values from the model used. Here you can find all the information regarding the models used and the similarity method used.
