import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from preprocess import preprocess_data, prepare_data

from xgboost import XGBClassifier
from ngboost import NGBClassifier
import numpy as np
from ngboost.distns import k_categorical
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# Preprocess and prepare data
df = preprocess_data('data/combined_reviews.csv')
X_train, X_test, y_train, y_test = prepare_data(df)



# Assuming `df['Description']` contains the text data after preprocessing
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
X = vectorizer.fit_transform(df['Description'])

# Save the vectorizer to disk
joblib.dump(vectorizer, 'models/vectorizer.pkl')

def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Validation Accuracy: {accuracy:.4f}')
    print(classification_report(y_test, y_pred,labels=[0, 1, 2], target_names=['Negative', 'Neutral', 'Positive']))

# Train the model
#rf_model = RandomForestClassifier()
#rf_model.fit(X_train, y_train)
y = df['Rating']

lr_model = LogisticRegression(max_iter=1000, C=8277.775528029204)
train_and_evaluate(lr_model,'Logistic Regression')

# Evaluate the model
#y_pred = rf_model.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#print('Random Forest Accuracy:', accuracy)

# Save the model and vectorizer
joblib.dump(lr_model, 'models/best_model.pkl')
# Save the vectorizer to disk
joblib.dump(vectorizer, 'models/vectorizer.pkl')