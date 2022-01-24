# DSND-Project2
### Summary
This is project 2 from Data Scientist Nanodegree, this project objective is to build a model that will be used by a web app to classify messages comes at a disaster timer.
### Table of Contnt:
1. [Runing instructions](#r-instructions)
2. [ETL pipeline](#etl)
3. [ML pipeline](#ml)
4. [Web app](web)

## Runing instructions <a name="r-instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## ETL pipeline <a name="etl"></a>
ETL pipeline is implemented in 'Data/process_data.py' and it applies the folowing:
1. load data from csv files and merge them into one data frame 
2. clean the new dataframe
3. save the new dataframe into sql database


## ML pipeline <a name="ml"></a>
ML pipeline is implemented in 'models/train_classifier.py' and it applies the folowing:
1. load dataframe from the database 
2. Do multiple text processing instruction
3. bulid the model and train it
4. evaluate the model by printing the classification report
5. save yhe model into Pickle file

## Web app<a name="web"></a>
Web app is implemented in 'app/run.py' and the code was provided from Udacity and the folowing changes were made:
1. change file paths for database and model
2. Add extra data visualizations 


