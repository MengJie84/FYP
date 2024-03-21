import pandas as pd
import glob
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import xgboost as xgb
# from keras.models import Sequential
# from keras.layers import Conv2D, Flatten, Dense

# Ensure nltk stopwords and wordnet are downloaded
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

def extract_participant_id(file_name):
    return int(re.match(r'(\d+)_', os.path.basename(file_name)).group(1))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

phq_scores = pd.read_csv('./fyp data/metadata_mapped.csv')
phq_scores['Participant_ID'] = phq_scores['Participant_ID'].astype(str).str.extract('(\d+)').astype(int)
folder_path_text = './fyp data/text'
folder_path_facial = './fyp data/facial'
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

    merged_data['PHQ_Label'] = merged_data['PHQ_Score'].apply(lambda x: x >= 10)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(merged_data['Processed_Text']).toarray()
    y = merged_data['PHQ_Label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print("Model Accuracy on test data:", test_accuracy * 100)

    full_predictions = model.predict(X)
    merged_data['Prediction'] = full_predictions
    full_accuracy = accuracy_score(merged_data['PHQ_Label'], merged_data['Prediction'])
    print("Full Model Accuracy on whole data:", full_accuracy * 100)

    predictions = pd.Series(full_predictions, index=merged_data['Participant_ID']).to_dict()
    print("Text Analysis Results:", predictions)

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

def decision_level_fusion(text_predictions, facial_predictions, actual_labels):
    fused_predictions = {}
    for participant_id, text_pred in text_predictions.items():
        facial_pred = facial_predictions.get(participant_id)
        if text_pred == facial_pred:
            fused_pred = text_pred
        else:
            fused_pred = text_pred

        fused_predictions[participant_id] = fused_pred
    return fused_predictions

if 'PHQ_Score' in phq_scores.columns:
    phq_scores['PHQ_Label'] = phq_scores['PHQ_Score'].apply(lambda x: x >= 10)
    actual_labels = phq_scores.set_index('Participant_ID')['PHQ_Label'].to_dict()
else:
    print("PHQ_Score column not found in phq_scores DataFrame.")
def print_confusion_matrix(predictions, actual_labels, title):
    TP = FP = TN = FN = 0
    for participant_id, predicted in predictions.items():
        actual = actual_labels.get(int(participant_id), None)
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

fused_predictions = decision_level_fusion(text_predictions, facial_predictions, actual_labels)
print_confusion_matrix(fused_predictions, actual_labels, "Fused Analysis")


# ------------------------models-------------------------------
data_items = fused_predictions.items()
data_list = list(data_items)
df = pd.DataFrame(data_list, columns=['Participant_ID', 'Prediction'])

df['Actual'] = df['Participant_ID'].apply(lambda x: actual_labels[x])

df['Prediction'] = df['Prediction'].astype(int)
df['Actual'] = df['Actual'].astype(int)

X = df[['Prediction']]
y = df['Actual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_model(model, X, y, kf):
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print(f"Mean CV Accuracy: {np.mean(cv_scores)}")
    print(f"CV Accuracy Scores: {cv_scores}")

print('--------------------XGBoost------------------------')
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
evaluate_model(xgb_model, X, y, kf)

print('\n---------------Ensemble Random Forest----------------------------')
rf_model = RandomForestClassifier(random_state=42)
evaluate_model(rf_model, X, y, kf)



# print('\n--------------CNN--------------------------------')
# accuracies = []
# f1_scores = []
# precisions = []
# recalls = []
#
# for train_index, test_index in kf.split(X):
#     # Splitting data
#     X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
#     y_train_fold, y_test_fold = y[train_index], y[test_index]
#
#     # Reshaping data for CNN
#     X_train_fold_reshaped = X_train_fold.values.reshape(X_train_fold.shape[0], 1, 1, 1)
#     X_test_fold_reshaped = X_test_fold.values.reshape(X_test_fold.shape[0], 1, 1, 1)
#     y_train_fold = y_train_fold.values
#     y_test_fold = y_test_fold.values
#
#     # Defining the CNN model
#     cnn_model = Sequential([
#         Conv2D(filters=64, kernel_size=(1, 1), activation='relu', input_shape=(1, 1, 1)),
#         Flatten(),
#         Dense(50, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#     cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     # Fitting the CNN model
#     cnn_model.fit(X_train_fold_reshaped, y_train_fold, epochs=10, batch_size=32, verbose=0)
#
#     # Evaluating the CNN model
#     predictions = (cnn_model.predict(X_test_fold_reshaped) > 0.5).astype("int32")
#     accuracies.append(accuracy_score(y_test_fold, predictions))
#     f1_scores.append(f1_score(y_test_fold, predictions))
#     precisions.append(precision_score(y_test_fold, predictions))
#     recalls.append(recall_score(y_test_fold, predictions))
#
# # Printing average scores across all folds
# print(f"Average Accuracy: {np.mean(accuracies)}")
# print(f"Average F1 Score: {np.mean(f1_scores)}")
# print(f"Average Precision: {np.mean(precisions)}")
# print(f"Average Recall: {np.mean(recalls)}")
