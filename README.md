# WorkPulse: Employee Review Sentiment Analysis

WorkPulse is a prototype for analyzing employee reviews. It includes scripts for predicting positive, neutral, or negative sentiment from review text, processing CSV uploads, and visualizing sentiment distributions and review terms.

## Repository contents

| Path | Purpose |
| --- | --- |
| `data/combined_reviews.csv` | Review dataset with 28,832 rows and the columns `Company`, `Position`, `Description`, and `Rating`. |
| `models/` | Pickled model and tokenizer artifacts used by the app experiments. |
| `scripts/app.py` | Taipy interface for single-review and CSV analysis, including sentiment summaries and visualizations. |
| `scripts/cleaning.py` | Text cleaning and feature preparation used by `scripts/app.py`. |
| `scripts/visualization.py` | Creates the sentiment pie chart and word cloud. |
| `scripts/attrition.ipynb` | Small, incomplete attrition notebook; it refers to `employee_attrition_prediction.csv`, which is not included. |
| `Extra Works/streamlit_app.py` | Alternative Streamlit interface for single reviews and CSV uploads. |
| `Extra Works/taipy_app.py` | Alternative Taipy interface and layout prototype. |
| `Extra Works/preprocess.py` | Preprocessing and data preparation functions used by the Streamlit and training scripts. |
| `Extra Works/preprocess1.py` | Separate sentiment prediction prototype. |
| `Extra Works/train_model.py` | Experimental model training script. |
| `Extra Works/styles.css` | Styles for the Taipy prototype. |
| `logo.png`, `wordcloud.png` | Image assets used by the interfaces and visualizations. |

## Sentiment labels

The preprocessing code maps review ratings to three classes:

| Rating | Class | Model label |
| --- | --- | --- |
| 3.5 and above | Positive | `2` |
| 2.5 or 3 | Neutral | `1` |
| 2 or below | Negative | `0` |

## Setup

Use Python and install dependencies for the interface you intend to run. The repository does not include a dependency lockfile or `requirements.txt`.

From the repository root, create and activate a virtual environment on Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

For the Taipy app and visualizations:

```powershell
pip install taipy pandas numpy nltk emoji keras-preprocessing lightgbm matplotlib wordcloud
```

For the Streamlit app, install its additional dependencies:

```powershell
pip install streamlit tensorflow scikit-learn joblib
```

The experimental training script also imports XGBoost and NGBoost:

```powershell
pip install xgboost ngboost
```

The preprocessing functions use NLTK corpora and taggers. Some scripts download NLTK resources when imported; if preprocessing reports a missing resource, install the resource named in the error with NLTK.

## Running the applications

Run commands from the repository root so paths to `data/` and `models/` resolve:

```powershell
python scripts/app.py
```

This starts the Taipy sentiment dashboard. To try the alternative Streamlit interface:

```powershell
streamlit run "Extra Works/streamlit_app.py"
```

The Streamlit interface accepts text for an individual prediction or a CSV upload. For CSV predictions, provide a `Description` column. The Taipy interface also accepts a CSV upload; it treats the first column as the review text.

## Training

The experimental training script reads `data/combined_reviews.csv` and writes model artifacts under `models/`:

```powershell
python "Extra Works/train_model.py"
```

Review the generated model and preprocessing artifacts before using them in an interface. The training script and app prototypes were developed as separate experiments and do not consistently share the same preprocessing or model pipeline.

## Known limitations

- The CSV preprocessing path in `Extra Works/preprocess.py` removes rows mapped to label `0`, so that path excludes negative reviews during training or batch preprocessing.
- The interfaces load different model files and use different preprocessing implementations. Their artifacts should be treated as experimental, and predictions should be validated before any operational use.
- `models/tfidf_vectorizer.pkl` is empty. Do not load it; the current apps do not use it.
- The attrition notebook expects an external CSV that is not present in this repository.
- Some paths and UI assets are specific to their prototype; the applications may need small path or pipeline adjustments when run in a different environment.
