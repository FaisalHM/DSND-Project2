import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    This fuction will load the data from data base
    using the given path
    
    
    Parameters:
    database_filepath (String): The path of the database 

    Returns:
    a (Dataframe): df contains only the message column
    b (Dataframe): df contains only the categories columns
    b.coulmn (List): list of all category names

    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine) 
    a = df['message']
    b = df.drop(['id', 'message', 'original', 'genre'], axis = 'columns')
    
    return a, b, b.columns


def tokenize(text):
    """
    This fuction will take the text and remove any thing but words, and 
    then tokenize the text and remove stop words ant then itrate throw 
    each token to lemmatize, lower, and strip it and finally add it to
    the clean_tokens in order to return it later.
    
    
    Parameters:
    text (String): The text to be processed
    
    Returns:
    clean_tokens (List): The list of tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    """
    This fuction will bulid the NLP pipline and create the 
    GridSearch eith parameters.
    
    
    Parameters:
    none
    
    Returns:
    cv (GridSearchCV): GridSearchCV object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__n_estimators': [10, 20],
                  'clf__estimator__min_samples_split': [2, 4]}
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This fuction will evaluate the model by printing
    the Classification Report
    
    
    Parameters:
    model
    X_test
    Y_test
    category_names
    
    Returns:
    None
    """
    y_pred = model.predict(X_test)
    print("Classification Report:\n")
    
    for category_index, category in enumerate(category_names):
        print(category)
        print(classification_report(Y_test[category], y_pred[:,category_index]))
    


def save_model(model, model_filepath):
    """
    This fuction will save the model in pickle file
    
    
    Parameters:
    model
    model_filepath (String): The path where the model will be saved
    
    Returns:
    None
    """
    picFile = open(model_filepath, 'wb')
    pickle.dump(model, picFile)
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()