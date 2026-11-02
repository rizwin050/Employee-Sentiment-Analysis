import streamlit as st
import joblib
import pandas as pd
from preprocess import preprocess_data
import numpy as np
import pickle


# Load the model
# Load the same vectorizer used during training
#with open('models/lgbm_model.pkl', 'rb') as f:
    #model = pickle.load(f)

model = joblib.load('models/lgbm_model.pkl')

class ReviewApp:
    def __init__(self):
        self.review_input = ""
        self.sentiment_output = ""
        self.upload_file = None
        self.positive_reviews = pd.DataFrame()
        self.negative_reviews = pd.DataFrame()
        self.neutral_reviews = pd.DataFrame()

    def predict_single_review(self):
        if self.review_input:
            preprocessed_review = preprocess_data(self.review_input)
            #review_data = self.prepare_review_data(preprocessed_review)
            prediction = model.predict(preprocessed_review)
            self.sentiment_output = "Positive" if prediction == 2 else "Negative" if prediction == 0 else "Neutral"

    def predict_multiple_reviews(self):
        if self.upload_file:
             # Preprocess the CSV data
            df = preprocess_data(self.upload_file)
            predictions = model.predict(df)
            
            # Create a DataFrame for displaying the results
            sentiment_df = pd.DataFrame({
                'Review': pd.read_csv(self.upload_file)['Description'],  # Original reviews
                'Sentiment': ["Positive" if pred == 2 else "Negative" if pred == 0 else "Neutral" for pred in predictions]
            })

            self.processed_reviews = sentiment_df
            # df = pd.read_csv(self.upload_file)
            # preprocessed_data = []
            # for review in df['Description']:
            #     processed_review = preprocess_data(review)
            #     preprocessed_data.append(processed_review)

            # # Convert the list of processed reviews into a NumPy array
            # preprocessed_data = np.vstack(preprocessed_data)
        
            # # Predict using the model
            # predictions = model.predict(preprocessed_data)
            # df['Sentiment'] = ["Positive" if pred == 2 else "Negative" if pred == 0 else "Neutral" for pred in predictions]

            # self.positive_reviews = df[df['Sentiment'] == 'Positive']
            # self.negative_reviews = df[df['Sentiment'] == 'Negative']
            # self.neutral_reviews = df[df['Sentiment'] == 'Neutral']

    #def prepare_review_data(self, review):
        # Convert the review to a 2D array
        #review_vectorized = vectorizer.transform([review])
        #return review_vectorized

    #def prepare_multiple_reviews_data(self, df):
        # Prepare data for multiple reviews (2D array)
        #return vectorizer.transform(df['Preprocessed_Description'])
    
    def build_ui(self):
        st.title("Review Sentiment Analysis")
        
        # Single Review Prediction
        self.review_input = st.text_area("Enter your review below:")
        if st.button("Predict"):
            self.predict_single_review()
            st.markdown(f"**Predicted Sentiment:** {self.sentiment_output}")

        self.upload_file = st.file_uploader("Upload a CSV file of reviews", type=["csv"])
        if self.upload_file:
            df=pd.read_csv(self.upload_file)
            st.write(df.head(5))
        if st.button("Predict for File"):
            self.predict_multiple_reviews()
            if self.processed_reviews is not None:
                st.dataframe(self.processed_reviews)

        # Bulk Review Prediction
        # self.upload_file = st.file_uploader("Upload a CSV file of reviews", type=["csv"])
        # if st.button("Predict for File"):
        #     self.predict_multiple_reviews()
        #     if not self.positive_reviews.empty:
        #         st.markdown("### Positive Reviews:")
        #         st.dataframe(self.positive_reviews)
        #     if not self.negative_reviews.empty:
        #         st.markdown("### Negative Reviews:")
        #         st.dataframe(self.negative_reviews)
        #     if not self.neutral_reviews.empty:
        #         st.markdown("### Neutral Reviews:")
        #         st.dataframe(self.neutral_reviews)


# Create and run the app
app = ReviewApp()
app.build_ui()
