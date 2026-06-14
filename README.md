# Sentiment Analysis on Amazon Alexa Customer Reviews

This project explores and analyzes customer reviews for Amazon Alexa devices, with the goal of building a sentiment classification model (positive vs. negative feedback) based on review text and product metadata.

## Dataset

**`amazon_alexa.csv`** — 3,150 customer reviews of Amazon Alexa products, with the following columns:

| Column              | Description                                              |
|---------------------|-----------------------------------------------------------|
| `rating`            | Star rating given by the customer (1–5)                  |
| `date`              | Date the review was submitted                            |
| `variation`         | Product variation/configuration (e.g. "Charcoal Fabric", "Black Dot") |
| `verified_reviews`  | The text content of the review                           |
| `feedback`          | Sentiment label — `1` (positive) or `0` (negative)        |

## Notebook

**`modeling and visualization part  .ipynb`** contains the analysis performed so far:

1. **Data loading & cleaning**
   - Loads `amazon_alexa.csv`, checks for and drops null values (1 record with a missing review).
2. **Exploratory data analysis (EDA)**
   - Distribution and percentage breakdown of `rating` values (bar chart & pie chart).
   - Distribution and percentage breakdown of `feedback` (positive ≈ 91.9%, negative ≈ 8.1%).
   - Relationship between `feedback` and `rating`.
   - Distribution and average rating per product `variation`.
   - Adds a `length` column (character length of each review) and visualizes its distribution overall and by feedback class.
3. **Text analysis**
   - Vectorizes review text with `CountVectorizer`.
   - Generates word clouds for all reviews, and for words unique to positive vs. negative reviews.

The notebook imports additional libraries for modeling (`RandomForestClassifier`, `DecisionTreeClassifier`, `XGBClassifier`, `GridSearchCV`, `StratifiedKFold`, `MinMaxScaler`, stemming with `PorterStemmer`, etc.), indicating the next steps are text preprocessing and training/evaluating classification models to predict `feedback` from `verified_reviews`.

## Sample Output Files

- **`SentimentBulk.csv`** — Example input file containing a list of sentences for bulk sentiment prediction.
- **`Predictions.csv`** — Example output showing predicted sentiment ("Positive"/"Negative") for a set of review sentences.

## Requirements

- Python 3
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `nltk` (with the `stopwords` corpus)
- `scikit-learn`
- `xgboost`
- `wordcloud`

## Usage

1. Install the required packages (e.g. `pip install numpy pandas matplotlib seaborn nltk scikit-learn xgboost wordcloud`).
2. Open `modeling and visualization part  .ipynb` in Jupyter Notebook/Lab.
3. Run the cells in order to reproduce the data cleaning, EDA, and word cloud visualizations.
