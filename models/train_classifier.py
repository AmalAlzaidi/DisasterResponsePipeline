import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    
    '''
    Load data from the SQLite database and split the dataset into X (features) and Y (target)
    
    INPUT - database_filepath - database file path
    
    OUTPUT - 
            X (features) and Y (target)
            
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    
    return X, Y


def tokenize(text):
    '''
    Strip punctuation , tokenize and lemmatize and remove stop words
    
    INPUT - text - massage text
    
    OUTPUT - 
            cleaned, tokenized and lemmatized text
            
    '''
    #strip punctuation 
    text = re.sub(r'[^\w\s]','', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    text = [w for w in words if w not in stopwords.words("english")]
    
    # lemmatize words
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    
    '''
    Build machine learning pipeline, trains and tunes the model using GridSearchCV
    
    
    
    OUTPUT - 
             GridSearchCV object
            
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {'clf__estimator__n_estimators': [10, 50, 100],
                  'clf__estimator__max_depth': [None,10, 50]
                 }
    
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    
    return cv

def evaluate_model(model, X_test, Y_test):
    
    '''
    Output results on the test set
    
    INPUT - model - the model to be evaluated
            X_test - feature's test data
            Y_test - label's test data
    
    OUTPUT - 
            print the f1 score, precision and recall for the test set for each category.
            
    '''
    y_pred = model.predict(X_test)
    for i, column in enumerate(Y_test):
        print(column)
        print(classification_report(Y_test[column], y_pred[:,i]))


def save_model(model, model_filepath):
    
    '''
    Save the final model as a pickle file
    
    INPUT - model - the model to be saved
            model_filepath - pickle file path
    
    OUTPUT - 
            final model as a pickle file.
            
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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