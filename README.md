# Fast-Furious-and-Insured-HackerEarth-Machine-Learning-Challenge

Problem
Vehicle insurance is insurance for cars, trucks, motorcycles, and other road vehicles. Its main purpose is to provide financial protection against:

Physical damage or bodily injury caused by traffic collisions
Liability that could arise from incidents in a vehicle
Vehicle insurance may additionally offer financial protection against theft of the vehicle and against damage to the vehicle sustained because of events other than traffic collisions such as keying, weather or natural disasters, and damage sustained by colliding with stationary objects.

# Task

You are required to perform the following tasks:

Condition: Predict if the vehicle provided in the image is damaged or not
Amount: Based on the condition of a vehicle, predict the insurance amount of the cars that are provided in the dataset
Data description

# The dataset folder contains the following:

The trainImages folder: Contains 1399 training images
The testImages folder: Contains 600 testing images
train.csv: Contains 1399 x 8 data points
test.csv: Contains 600 x 6 data points
sample_submission.csv: Contains 5 x 3 data points
The columns in the dataset are as follows:

Column name	Description
Image_path	Represents the name of an image 
Insurance_company	Represents masked values of some insurance companies
Cost_of_vehicle	Represents the cost of a vehicle present in the image
Min_coverage	Represents the minimum coverage provided by an insurance company
Expiry_date	Represents the expiry date of the insurance
Max_coverage	Represents the minimum coverage provided by an insurance company
Condition	Represents whether a vehicle is damaged
Amount	Represents the insurance amount of a vehicle
Evaluation metric
# For predictions of the Condition column
score1 = max(0, 100*metrics.f1_score(actualConditions, predictedConditions, average="micro"))

# For predictions of the Amount column
score2 = max(0, 100*metrics.r2_score(actualAmount, predictedAmount))

final_score = (score1/2)+(score2/2)
Result submission guidelines
The indexes are Image_path. 
The target is the Condition and Amount column. 
The submission file must be submitted in .csv format only.
The size of this submission file must be 600 x 3.
Note: Ensure that your submission file contains the following:

Correct index values as per the test file
Correct names of columns as provided in the sample_submission.csv file

# Link of Challenge:
https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-vehicle-insurance-claim/
