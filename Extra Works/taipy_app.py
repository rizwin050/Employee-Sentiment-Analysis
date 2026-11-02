import taipy.gui as tp
import joblib
import pandas as pd
from preprocess import preprocess_data

# Load the model
model = joblib.load('models/best_model.pkl')

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
            review_data = self.prepare_review_data(preprocessed_review)
            prediction = model.predict(review_data)
            self.sentiment_output = "Positive" if prediction == 2 else "Negative" if prediction == 0 else "Neutral"

    def predict_multiple_reviews(self):
        if self.upload_file:
            df = pd.read_csv(self.upload_file)
            df['Preprocessed_Description'] = df['Description'].apply(preprocess_data)
            review_data = self.prepare_multiple_reviews_data(df)
            predictions = model.predict(review_data)
            df['Sentiment'] = ["Positive" if pred == 2 else "Negative" if pred == 0 else "Neutral" for pred in predictions]

            self.positive_reviews = df[df['Sentiment'] == 'Positive']
            self.negative_reviews = df[df['Sentiment'] == 'Negative']
            self.neutral_reviews = df[df['Sentiment'] == 'Neutral']

    def prepare_review_data(self, review):
        # Prepare data for single review (dummy implementation)
        return [review]

    def prepare_multiple_reviews_data(self, df):
        # Prepare data for multiple reviews (dummy implementation)
        return df['Preprocessed_Description'].values

    def build_ui(self):
        page = tp.Page(
            tp.Markdown("### Review Sentiment Analysis"),
            tp.Markdown("**Enter your review below:**"),
            tp.Textarea(value=tp.bind(self, 'review_input'), rows=4, cols=50),
            tp.Button("Predict", on_click=self.predict_single_review),
            tp.Markdown("**Predicted Sentiment:**"),
            tp.Markdown(tp.bind(self, 'sentiment_output')),

            tp.Markdown("**Upload a CSV file of reviews:**"),
            tp.FileUpload(label="Upload a CSV file", on_change=tp.bind(self, 'upload_file')),
            tp.Button("Predict for File", on_click=self.predict_multiple_reviews),

            tp.Markdown("### Positive Reviews:"),
            tp.DataTable(dataframe=tp.bind(self, 'positive_reviews')),
            tp.Markdown("### Negative Reviews:"),
            tp.DataTable(dataframe=tp.bind(self, 'negative_reviews')),
            tp.Markdown("### Neutral Reviews:"),
            tp.DataTable(dataframe=tp.bind(self, 'neutral_reviews'))
        )
        return page


# Create and run the app
app = ReviewApp()
gui = tp.Gui(app.build_ui())
gui.run(use_reloader=True)