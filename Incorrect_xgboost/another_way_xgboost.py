import pandas as pd
import glob
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
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

def decision_fusion(text_predictions, facial_predictions):
    fused_predictions = {}
    all_participants = set(text_predictions.keys()) | set(facial_predictions.keys())

    for participant_id in all_participants:
        text_pred = text_predictions.get(participant_id, False)  # Defaults to False if not found
        facial_pred = facial_predictions.get(participant_id, False)  # Defaults to False if not found
        fused_predictions[participant_id] = text_pred or facial_pred

    return fused_predictions

# def weighted_decision_fusion(text_predictions, facial_predictions, text_weight=0.6, facial_weight=0.4):
#     fused_predictions = {}
#     all_participants = set(text_predictions.keys()) | set(facial_predictions.keys())
#
#     for participant_id in all_participants:
#         text_pred = text_predictions.get(participant_id, 0)  # Assuming these are now confidence scores
#         facial_pred = facial_predictions.get(participant_id, 0)  # Assuming these are now confidence scores
#         weighted_score = text_pred * text_weight + facial_pred * facial_weight
#         fused_predictions[participant_id] = weighted_score >= 0.5  # Adjust the threshold as necessary
#
#     return fused_predictions


# Execute the fusion
fused_predictions = decision_fusion(text_predictions, facial_predictions)

# Print Fused Predictions
# for participant_id, prediction in fused_predictions.items():
#     print(f"Participant ID: {participant_id}, Fused Prediction: {'Depressed' if prediction else 'Not Depressed'}")

print('-----------------------XGBoost------------------------')
# Example dictionaries
text_features_dict = {'1': True, '2': False, '3': True}
facial_features_dict = {'1': False, '2': True, '3': False}

# Assuming text_predictions and facial_predictions are your source dictionaries
text_features_df = pd.DataFrame(list(text_predictions.items()), columns=['Participant_ID', 'Text_Feature'])
facial_features_df = pd.DataFrame(list(facial_predictions.items()), columns=['Participant_ID', 'Facial_Feature'])
fused_labels_df = pd.DataFrame(list(fused_predictions.items()), columns=['Participant_ID', 'Fused_Label'])

# Your existing setup for data preparation remains unchanged

# Convert 'Participant_ID' to string in all DataFrames to ensure consistency
text_features_df['Participant_ID'] = text_features_df['Participant_ID'].astype(str)
facial_features_df['Participant_ID'] = facial_features_df['Participant_ID'].astype(str)
fused_labels_df['Participant_ID'] = fused_labels_df['Participant_ID'].astype(str)

# Merge the features DataFrames and then the fused labels
features_df = pd.merge(text_features_df, facial_features_df, on='Participant_ID', how='outer')
features_df = pd.merge(features_df, fused_labels_df, on='Participant_ID', how='inner')

# Splitting features and labels
X = features_df[['Text_Feature', 'Facial_Feature']]  # Assuming these are the feature columns
y = features_df['Fused_Label'].astype(int)  # Convert True/False to 1/0

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize XGBoost classifier with some basic parameters
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    max_depth=3,
    n_estimators=100,
    random_state=42
)

# Perform cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = cross_val_score(xgb_model, X_train, y_train, cv=kfold, scoring='accuracy')

# Output the mean and standard deviation of the cross-validated scores
print(f"CV Accuracy: {cv_results.mean():.2f} with standard deviation {cv_results.std():.2f}")

# Train the model on the training set
xgb_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = xgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Test Set Accuracy: {test_accuracy:.2f}")

