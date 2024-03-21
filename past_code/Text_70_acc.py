import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import glob
import os

nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load stopwords
stopwords = set(nltk.corpus.stopwords.words('english'))

folder_path = '../fyp data/text'
all_files_text = glob.glob(folder_path + '/*.csv')
label = pd.read_csv('../fyp data/metadata_mapped.csv')


# Extract Participant ID from filename
def extract_participant_id(file_name):
    return os.path.basename(file_name).split('_')[0]


# Initialize an empty list to store results
results = []

# Iterate through each text file
for file in all_files_text:
    # Extract Participant ID from the filename
    participant_id = extract_participant_id(file)

    # Load text data
    df = pd.read_csv(file)
    text_data = df['Text']

    # Filter label data based on Participant ID
    file_labels = label[label['Participant_ID'] == int(participant_id)]

    # Check if the label indicates depression
    if len(file_labels) > 0 and file_labels.iloc[0]['PHQ_Score'] >= 10:
        depression_detected = True
    else:
        depression_detected = False

    # Perform sentiment analysis
    lowercase_text_data = text_data.str.lower()  # Convert to lower case
    cleaned_text_data = lowercase_text_data.apply(
        lambda x: ' '.join([word for word in x.split() if word not in stopwords and word.isalpha()]))

    sid = SentimentIntensityAnalyzer()
    sentiment_scores = cleaned_text_data.apply(sid.polarity_scores)

    # Determine if depression is detected based on sentiment scores
    overall_sentiment_score = sentiment_scores.apply(lambda x: x['compound']).mean()
    if overall_sentiment_score <= -1:
        depression_detected_sentiment = True
    else:
        depression_detected_sentiment = False

    # Append results to the list
    results.append({
        'Participant_ID': participant_id,
        'Depression_Label': depression_detected,
        'Depression_Sentiment': depression_detected_sentiment
    })

# Print or further process the results
for res in results:
    print(
        f"Participant ID: {res['Participant_ID']}, Depression Detected (Label): {res['Depression_Label']}, Depression Detected (Sentiment Analysis): {res['Depression_Sentiment']}")


# Calculate accuracy
true_positives = sum(1 for res in results if res['Depression_Label'] and res['Depression_Sentiment'])
true_negatives = sum(1 for res in results if not res['Depression_Label'] and not res['Depression_Sentiment'])
total_samples = len(results)

accuracy = ((true_positives + true_negatives) / total_samples)*100

print("Accuracy:", accuracy, '%')



