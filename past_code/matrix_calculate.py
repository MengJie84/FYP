import pandas as pd
import glob
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Ensure nltk stopwords and wordnet are downloaded
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

def extract_participant_id(file_name):
    return int(re.match(r'(\d+)_', os.path.basename(file_name)).group(1))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load PHQ scores
phq_scores = pd.read_csv('../fyp data/metadata_mapped.csv')
phq_scores['Participant_ID'] = phq_scores['Participant_ID'].astype(str).str.extract('(\d+)').astype(int)
folder_path_text = '../fyp data/text'
folder_path_facial = '../fyp data/facial'
all_files_text = glob.glob(os.path.join(folder_path_text, '*.csv'))
all_files_facial = glob.glob(folder_path_facial + '/*.csv')
relevant_au = ['AU01_c', 'AU04_c', 'AU06_c', 'AU12_c', 'AU15_c', 'AU17_c']
def preprocess_text(text, lemmatizer, stop_words):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def text_analysis(all_files, phq_scores):
    print("----------------------Text Analysis------------------------")
    participant_text_data = []
    for file in all_files:
        participant_id = extract_participant_id(file)
        temp_df = pd.read_csv(file)
        all_text = ' '.join(temp_df.fillna('').astype(str).sum(axis=0).tolist())
        processed_text = preprocess_text(all_text, lemmatizer, stop_words)
        participant_text_data.append({'Participant_ID': participant_id, 'Processed_Text': processed_text})

    text_data = pd.DataFrame(participant_text_data)
    merged_data = text_data.merge(phq_scores, on='Participant_ID')

    # Ensure PHQ_Label is created based on PHQ scores before using it
    merged_data['PHQ_Label'] = merged_data['PHQ_Score'].apply(lambda x: x >= 10)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(merged_data['Processed_Text']).toarray()
    y = merged_data['PHQ_Label'].values

    # Split the dataset for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print("Model Accuracy on test data:", test_accuracy * 100)

    # Predict for the entire dataset to label each participant
    full_predictions = model.predict(X)
    merged_data['Prediction'] = full_predictions
    full_accuracy = accuracy_score(merged_data['PHQ_Label'], merged_data['Prediction'])
    print("Full Model Accuracy on whole data:", full_accuracy * 100)

    predictions = pd.Series(full_predictions, index=merged_data['Participant_ID']).to_dict()
    print("Text Analysis Results:", predictions)

    # Return predictions and the full accuracy
    return predictions


def analyze_depression_via_facial_cues(all_files_facial, phq_scores, relevant_au):
    print("-------------------------Facial Analysis-------------------------")
    threshold = 10
    depressed_frequencies = []
    non_depressed_frequencies = []
    for file in all_files_facial:
        participant_id = extract_participant_id(file)
        df = pd.read_csv(file)
        if participant_id in phq_scores['Participant_ID'].values:
            file_labels = phq_scores[phq_scores['Participant_ID'] == participant_id]
            total_frequency = sum(df[au].sum() for au in relevant_au)
            if file_labels.iloc[0]['PHQ_Score'] >= threshold:
                depressed_frequencies.append(total_frequency)
            else:
                non_depressed_frequencies.append(total_frequency)

    avg_depressed_frequency = sum(depressed_frequencies) / len(depressed_frequencies) if depressed_frequencies else 0
    avg_non_depressed_frequency = sum(non_depressed_frequencies) / len(non_depressed_frequencies) if non_depressed_frequencies else 0

    depression_detection_results = {}
    correct_predictions = 0
    for file in all_files_facial:
        participant_id = extract_participant_id(file)
        df = pd.read_csv(file)
        if participant_id in phq_scores['Participant_ID'].values:
            total_frequency = sum(df[au].sum() for au in relevant_au)
            depression_detected = total_frequency > avg_non_depressed_frequency
            depression_detection_results[participant_id] = depression_detected
            actual_depression = phq_scores[phq_scores['Participant_ID'] == participant_id].iloc[0]['PHQ_Score'] >= threshold
            correct_predictions += int(depression_detected == actual_depression)

    total_participants = len(depression_detection_results)
    accuracy = (correct_predictions / total_participants) * 100 if total_participants > 0 else 0
    print("Facial Analysis Results:", depression_detection_results)
    print(f"Facial Analysis Accuracy: {accuracy}%")

    return depression_detection_results

text_predictions = text_analysis(all_files_text, phq_scores)
facial_predictions = analyze_depression_via_facial_cues(all_files_facial, phq_scores, relevant_au)


if 'PHQ_Score' in phq_scores.columns:
    # Create 'PHQ_Label' to indicate depression based on 'PHQ_Score' >= 10
    phq_scores['PHQ_Label'] = phq_scores['PHQ_Score'].apply(lambda x: x >= 10)
    # Convert 'PHQ_Label' into a dictionary for easy lookup
    actual_labels = phq_scores.set_index('Participant_ID')['PHQ_Label'].to_dict()
else:
    print("PHQ_Score column not found in phq_scores DataFrame.")

def print_confusion_matrix(predictions, actual_labels, title):
    TP = FP = TN = FN = 0
    for participant_id, predicted in predictions.items():
        participant_id = int(participant_id)
        actual = actual_labels.get(participant_id, None)
        if actual is not None:
            if predicted == True:
                if actual == True:
                    TP += 1
                else:
                    FP += 1
            else:
                if actual == False:
                    TN += 1
                else:
                    FN += 1
        else:
            print(f"Warning: Participant_ID {participant_id} not found in actual labels.")

    print(f"{title} Confusion Matrix:")
    print(f"TP: {TP}, FP: {FP}")
    print(f"FN: {FN}, TN: {TN}\n")

print_confusion_matrix(text_predictions, actual_labels, "Text Analysis")
print_confusion_matrix(facial_predictions, actual_labels, "Facial Analysis")