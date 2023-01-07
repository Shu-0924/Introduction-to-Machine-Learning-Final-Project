# ML Final Project
You can find the Kaggle competition of this final project from the following link:  <br>
https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview

<br>

## Environment & Requirements
### Device:
> Training the task on GPU (GTX 1650) about 3 minutes in total.
### Python version:
> Python 3.10.4
### Packages required:
    keras==2.9.0
    numpy==1.22.4
    tensorflow==2.9.1
(Other packages will be installed when you install these three packages)
### To install requirements:
    pip install -r requirements.txt

<br>

## Dataset
All the training and testing data are from  <br>
https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data

<br>

## Training
To train the model, run 109550074_Final_train.py or download the model I 
have already trained from the following links.  <br>
https://drive.google.com/file/d/1KmXxU3BVjoGId9q-ekXHF2ZQaCDW9jK0/view?usp=share_link

<br>

## Evaluation
To test the result of the model, run 109550074_Final_inference.py

Please remember to modify the path of "test.csv" and "model.h5"

<br>

## Results
The average accuracy is based on 5 submission without selection after training

| Model       |  Best Accuracy   | Average Accuracy |
| ----------- |----------------- | ---------------- |
| My model    |      0.5906      |     0.590294     |

![](https://i.imgur.com/jBUCvDY.png)

<br>

## Reproducing Submission
To reproduct my submission without retraining, do the following steps:

1. Download 109550074_Final_inference.py from here
    
2. Download model.h5 from the link I provide and put it into the same folder with 109550074_Final_inference.py  <br>
   (Or put anywhere you want but remember to modify the path to model.h5 in 109550074_Final_inference.py)

3. Download test.csv from Kaggle or here, and put it into the same folder with 109550074_Final_inference.py  <br>
   (Or put anywhere you want but remember to modify the path to test.csv in 109550074_Final_inference.py)

4. Run the following command (use your path of python 3.10)

        virtualenv -p <path/to/python3.10> myenv

5. Put requirement.txt into the same folder with 109550074_Final_inference.py and run command

         pip install -r requirements.txt
         
6. Run following command to start inference
        
        source myenv/bin/activate
        python 109550074_Final_inference.py
        
7. You can see the "submission.csv" in the same folder with 109550074_Final_inference.py, you can now submit it to Kaggle.
