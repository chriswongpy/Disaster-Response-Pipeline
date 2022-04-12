
### Table of Contents

1. [Project Overview](#overview)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview <a name="overview"></a>
In this project, we will analyze disaster data from Appen (formally Figure 8) and build a model for an API that classifies disaster messages.

There are three components in this project.

1. ETL Pipeline - to load and prepare the data
2. ML Pipeline - to build the machine learning pipeline, train the model and export the final model as a pickle file  
3. Flask Web App - below are some screenshots
![image](https://user-images.githubusercontent.com/103321006/162997487-19d56c8a-0dd7-4b3c-9892-6f3d4b43b27e.png)


## Installation <a name="installation"></a>
The code should run with no issues using Python versions 3.*, with libraries of numpy, pandas, sqlalchemy, NLTK, re, pickle, Sklearn, plotly and flask libraries.

## File Descriptions <a name="files"></a>
1. "app" folder including the templates folder and "run.py" for the web application
2. "data" folder containing "DisasterResponse.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py" for data preparation
3. "model" folder including "classifier.pkl" (too big to store here but can follow the [instructions](#instructions) below to build) and "train_classifier.py" for the Machine Learning model.
4. README file


## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory and run your web app: `python run.py`
3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

credit to [Appen](https://www.figure-eight.com/) (formally Figure 8) for providing the data for this training.  
