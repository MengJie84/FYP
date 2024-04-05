import pandas as pd
import glob
import os
import re
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Input

# Ensure nltk stopwords and wordnet are downloaded
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

def extract_participant_id(file_name):
    return int(re.match(r'(\d+)_', os.path.basename(file_name)).group(1))

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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
    # print('Length: ', len(predictions))

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
    # print('length: ', len(depression_detection_results))

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

        # print('length: ',  len(fused_predictions))
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


# Convert fused predictions and actual labels to DataFrame
data_items = list(fused_predictions.items())
df = pd.DataFrame(data_items, columns=['Participant_ID', 'Prediction'])
df['Actual'] = df['Participant_ID'].map(actual_labels)
def plot_matrix(df, title):
    cm = confusion_matrix(df['Actual'], df['Prediction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Not Depressed', 'Predicted Depressed'],
                yticklabels=['Actual Not Depressed', 'Actual Depressed'])
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

X = df[['Prediction']].values  # This might change if you have more features
y = df['Actual'].values

random_under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)  # 'majority' can also be used depending on the desired strategy

X_resampled, y_resampled = random_under_sampler.fit_resample(X, y)

df_resampled = pd.DataFrame(X_resampled, columns=['Prediction'])
df_resampled['Actual'] = y_resampled

# Plot the matrix before and after balancing
plot_matrix(df, 'Matrix Before Balancing')
plot_matrix(df_resampled, 'Matrix After Balancing')

# Preparing data for model training with the resampled balanced data
X_balanced = df_resampled[['Prediction']].values  # Features from resampled balanced DataFrame
y_balanced = df_resampled['Actual'].values

class_count_before = Counter(df['Actual'])
print("Before balance: ", class_count_before)
class_counts_after = Counter(y_balanced)
print("After balance: ", class_counts_after)

# Split balanced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# ------------------------models-------------------------------
def evaluate_model(model, X, y, kf):
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print(f"Mean CV Accuracy: {np.mean(cv_scores)}")
    print(f"CV Accuracy Scores: {cv_scores}")

    predictions = cross_val_predict(model, X, y, cv=kf)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Evaluate XGBoost
print('--------------------XGBoost------------------------')
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
evaluate_model(xgb_model, X_balanced, y_balanced, kf)

# Evaluate Random Forest
print('\n---------------Ensemble Random Forest----------------------------')
rf_model = RandomForestClassifier(random_state=24)
evaluate_model(rf_model, X_balanced, y_balanced, kf)


print('\n--------------CNN--------------------------------')
scaler = StandardScaler()
X = scaler.fit_transform(df[['Prediction']].values)  # Adjust if more features
y = df['Actual'].values

X_reshaped = X.reshape((-1, 1, 1, 1))

# Define the CNN model structure outside the loop
def create_cnn_model():
    model = Sequential([
        Input(shape=(1, 1, 1)),
        Conv2D(32, kernel_size=(1, 1), activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Use 'softmax' for multiclass classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

accuracies = []
precisions = []
recalls = []
f1_scores = []
# Iterate over each fold
for fold, (train_index, test_index) in enumerate(kf.split(X_reshaped), start=1):
    X_train_fold, X_test_fold = X_reshaped[train_index], X_reshaped[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    model = create_cnn_model()

    print(f'Training on fold {fold}...')
    model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

    predictions = (model.predict(X_test_fold) > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test_fold, predictions)
    precision = precision_score(y_test_fold, predictions)
    recall = recall_score(y_test_fold, predictions)
    f1 = f1_score(y_test_fold, predictions)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f"Fold {fold} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
