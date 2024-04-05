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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

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

# Plot the matrix before and after balancing
plot_matrix(df, 'Matrix Before Balancing')

class_count_before = Counter(df['Actual'])
print("Before balance: ", class_count_before)

# Split balanced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# ------------------------models-------------------------------
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    # print(f"Mean CV Accuracy: {np.mean(cv_scores)}")
    # print(f"CV Accuracy Scores: {cv_scores}")
    results['Model'].append(model_name)
    results['Accuracy'].append(predictions)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)
    print(f"{model_name} - Accuracy: {predictions}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


# Evaluate XGBoost
print('--------------------XGBoost------------------------')
evaluate_model('XGBoost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),X_train, X_test, y_train, y_test)


# Evaluate Random Forest
print('\n---------------Ensemble Random Forest----------------------------')
evaluate_model('Random Forest', RandomForestClassifier(random_state=24), X_train, X_test, y_train, y_test)

print('\n---------------Logistic Regression----------------------------')
logistic_model = LogisticRegression(solver='liblinear', random_state=42)
evaluate_model('Logistic Regression', logistic_model, X_train, X_test, y_train, y_test)

print('\n---------------Decision Tree----------------------------')
tree_model = DecisionTreeClassifier(random_state=42)
evaluate_model('Decision Tree', tree_model,X_train, X_test, y_train, y_test)

print('\n---------------Naive Bayes----------------------------')
nb_model = GaussianNB()
evaluate_model('Naive Bayes', nb_model, X_train, X_test, y_train, y_test)

print('\n---------------K-NN----------------------------')
knn_model = KNeighborsClassifier()
evaluate_model('k-NN', knn_model, X_train, X_test, y_train, y_test)

print('\n--------------CNN--------------------------------')

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

X = df.drop('Actual', axis=1).values  # Adjust if your features are named differently
y = df['Actual'].values

# Scale your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN input if necessary
X_reshaped = X_scaled.reshape((-1, 1, 1, 1))
y_reshaped = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)
model = create_cnn_model()
print("Training the CNN model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict on the test set
predictions = (model.predict(X_test) > 0.5).astype(int).flatten()

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Add the CNN mean scores to the results dictionary
results['Model'].append('CNN')
results['Accuracy'].append(accuracy)
results['Precision'].append(precision)
results['Recall'].append(recall)
results['F1 Score'].append(f1)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

csv_generate_path = '../fyp data/imbalance.csv'
df_results.to_csv(csv_generate_path, index=False)
print(f"Sucess save result to {csv_generate_path}")