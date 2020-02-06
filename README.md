
# Disaster Response Pipeline

## Table of Contents
1. [Installation](#Installation)
2. [Project Motivation](#Project-Motivation)
3. [File Description](#File-Description)
4. [Acknowledgements](#Acknowledgements)


### Installation
- Python 3.5+
- Python libraries: numpy, pandas.
- Machine Learning Libraries: Siki-Learn
- Natural Languge Process Libraries: NLTK
- SQLlite Database: SQLAlchemy
- Web App: Flask

### Project Motivation
In this project, I built a machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.
### File Description
- app
  - template
    - master.html: main page of web app
    - go.html: classification result page of web app
  - run.py: Flask file that runs app

- data
  - disaster_categories.csv: data to process 
  - disaster_messages.csv: data to process
  - process_data.py
  - InsertDatabaseName.db: database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl: saved model 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements
Must give credit to Figure Eight for the data.
