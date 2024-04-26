import pandas as pd
import glob
import os
import re
import seaborn as sns
from imblearn.combine import SMOTETomek
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE, RandomOverSampler
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

phq_scores = pd.read_csv('../fyp data/metadata_mapped.csv')
phq_scores['Participant_ID'] = phq_scores['Participant_ID'].astype(str).str.extract('(\d+)').astype(int)
folder_path_text = '../fyp data/text'
folder_path_facial = '../fyp data/facial'
all_files_text = glob.glob(os.path.join(folder_path_text, '*.csv'))
all_files_facial = glob.glob(folder_path_facial + '/*.csv')
relevant_au = ['AU01_c', 'AU04_c', 'AU06_c', 'AU12_c', 'AU15_c', 'AU17_c']

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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
    print("Model Accuracy on test my_data.json:", test_accuracy * 100)

    full_predictions = model.predict(X)
    merged_data['Prediction'] = full_predictions
    full_accuracy = accuracy_score(merged_data['PHQ_Label'], merged_data['Prediction'])
    print("Full Model Accuracy on whole my_data.json:", full_accuracy * 100)

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

        df.replace('', pd.NA, inplace=True)
        df.dropna(subset=relevant_au, inplace=True)
        df[relevant_au] = df[relevant_au].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=relevant_au, inplace=True)

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


data_items = list(fused_predictions.items())
df = pd.DataFrame(data_items, columns=['Participant_ID', 'Prediction'])
df['Actual'] = df['Participant_ID'].map(actual_labels)
def plot_matrix(df, title, file_path):
    df['Prediction'] = (df['Prediction'] > 0.5).astype(int)
    cm = confusion_matrix(df['Actual'], df['Prediction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Not Depressed', 'Predicted Depressed'],
                yticklabels=['Actual Not Depressed', 'Actual Depressed'])
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(file_path)
    plt.show()

X = df[['Prediction']].values
y = df['Actual'].values
X = X.astype(float)

hybrid_sampler = SMOTETomek(sampling_strategy='auto', random_state=42)  # 'majority' can also be used depending on the desired strategy
X_resampled, y_resampled = hybrid_sampler.fit_resample(X, y)

scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

df_resampled = pd.DataFrame(X_resampled, columns=['Prediction'])
df_resampled['Actual'] = y_resampled

# Plot the matrix before and after balancing
plot_matrix(df, 'Matrix Before Balancing', 'matrix_before_balancing.png')
plot_matrix(df_resampled, 'Matrix After Balancing', 'matrix_after_balancing.png')

# Preparing my_data.json for model training with the resampled balanced my_data.json
X_balanced = df_resampled[['Prediction']].values  # Features from resampled balanced DataFrame
y_balanced = df_resampled['Actual'].values

class_count_before = Counter(df['Actual'])
print("Before balance: ", class_count_before)
class_counts_after = Counter(y_balanced)
print("After balance: ", class_counts_after)

# Split balanced my_data.json into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# ------------------------models-------------------------------
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'Error Rate': [],
    'Specificity': [],
    'AUC-ROC': [],
    'False Positive Rate': []
}

cnn_metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Error Rate': [],
        'Specificity': [],
        'AUC-ROC': [],
        'False Positive Rate': []
    }

all_runs_results = {
    'XGBoost': [],
    'Random Forest': [],
    'Logistic Regression': [],
    'CNN': []
}

def calculate_additional_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    error_rate = (fp + fn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    fpr = fp / (tn + fp)
    return error_rate, specificity, fpr

def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, kf):
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro', zero_division=0)
    recall = recall_score(y_test, predictions, average='macro', zero_division=0)
    f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    error_rate, specificity, fpr = calculate_additional_metrics(y_test, predictions)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC-ROC': auc_roc,
        'Error Rate': error_rate,
        'Specificity': specificity,
        'False Positive Rate': fpr
    }

    return metrics

def evaluate_model_with_thresholds(model_name, model, X_train, y_train, X_test, y_test, thresholds):
    model.fit(X_train, y_train)
    y_probs = model.predict(X_test).ravel()  # For binary classification

    # Dictionary to accumulate metrics for each threshold
    threshold_metrics = {'Accuracy': [], 'Recall': [], 'Specificity': [], 'AUC-ROC': [], 'False Positive Rate': [], 'Precision': [] }

    auc_roc = roc_auc_score(y_test, y_probs)
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)


        tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (tn + fp)

        threshold_metrics['Accuracy'].append(accuracy)
        threshold_metrics['Recall'].append(recall)
        threshold_metrics['Specificity'].append(specificity)
        threshold_metrics['AUC-ROC'].append(auc_roc)
        threshold_metrics['False Positive Rate'].append(fpr)
        threshold_metrics['Precision'].append(precision)

    return threshold_metrics

def evaluate_model_with_thresholds_cnn(model, X_test, y_test, thresholds):
    X_test_reshaped = X_test.reshape(-1, 1, 1, 1)
    probabilities = model.predict(X_test_reshaped).flatten()

    threshold_metrics = {'Accuracy': [], 'Recall': [], 'Specificity': [], 'AUC-ROC': [], 'False Positive Rate': [],
                         'Precision': []}
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)

        tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (tn + fp)

        threshold_metrics['Accuracy'].append(accuracy)
        threshold_metrics['Recall'].append(recall)
        threshold_metrics['Specificity'].append(specificity)
        threshold_metrics['AUC-ROC'].append(auc_roc)
        threshold_metrics['False Positive Rate'].append(fpr)
        threshold_metrics['Precision'].append(precision)

    return threshold_metrics

def create_cnn_model():
    model = Sequential([
        Input(shape=(1, 1, 1)),
        Conv2D(32, kernel_size=(1, 1), activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

n_runs = 2
models = [
    ('XGBoost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('Random Forest', RandomForestClassifier(random_state=24)),
    ('Logistic Regression', LogisticRegression(solver='liblinear', random_state=42))
]

cnn_metrics_list = []
model_conf_matrices = {model_name: [] for model_name, _ in models}
for run in range(n_runs):
    print(f"Run {run+1}/{n_runs}")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for model_name, model in models:
        metrics = evaluate_model(model_name, model, X_train, y_train, X_test, y_test, kf)
        all_runs_results[model_name].append(metrics)
        y_probs = model.predict_proba(X_test)[:, 1]

        print('\n--------------CNN--------------------------------')
        scaler = StandardScaler()
        X = scaler.fit_transform(X_balanced)
        X_reshaped = X.reshape((-1, 1, 1, 1))

        for fold, (train_index, test_index) in enumerate(kf.split(X_reshaped), start=1):
            X_train_fold, X_test_fold = X_reshaped[train_index], X_reshaped[test_index]
            y_train_fold, y_test_fold = y_balanced[train_index], y_balanced[test_index]

            model = create_cnn_model()
            print(f'Training on fold {fold}...')
            model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)
            predictions = (model.predict(X_test_fold) > 0.5).astype(int).flatten()
            probabilities = model.predict(X_test_fold).flatten()

            accuracy = accuracy_score(y_test_fold, predictions)
            precision = precision_score(y_test_fold, predictions)
            recall = recall_score(y_test_fold, predictions)
            f1 = f1_score(y_test_fold, predictions)
            tn, fp, fn, tp = confusion_matrix(y_test_fold, predictions).ravel()
            error_rate = (fp + fn) / (tp + tn + fp + fn)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            auc_roc = roc_auc_score(y_test_fold, probabilities)

            cnn_metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Error Rate': error_rate,
                'Specificity': specificity,
                'False Positive Rate': fpr,
                'AUC-ROC': auc_roc
            }
            cnn_metrics_list.append(cnn_metrics)


# print(f"CNN Final Mean Metrics Across Runs: {cnn_metrics_list}")

final_mean_results = {
    model_name: {
        metric: np.mean([run[metric] for run in all_runs_results[model_name]])
        for metric in all_runs_results[model_name][0]
    }
    for model_name, _ in models
}
if cnn_metrics_list:
    cnn_final_mean_metrics = {
        metric: np.mean([metrics[metric] for metrics in cnn_metrics_list])
        for metric in cnn_metrics_list[0]
    }
    final_mean_results['CNN'] = cnn_final_mean_metrics
else:
    print("No CNN metrics to aggregate.")

print(final_mean_results)

df_data = []
for model_name, metrics in final_mean_results.items():
    row = {'Model': model_name}
    row.update(metrics)
    df_data.append(row)

df_results = pd.DataFrame(df_data)

print(df_results.to_string(index=False))

csv_generate_path = '../fyp data/hybrid_sampling.csv'
df_results.to_csv(csv_generate_path, index=False)
print(f"Results saved to {csv_generate_path}")

thresholds = np.linspace(0, 1, 100)
cnn_model = create_cnn_model()
cnn_metrics = evaluate_model_with_thresholds_cnn(cnn_model, X_test, y_test, thresholds)
model_threshold_metrics = {}
model_threshold_metrics['CNN'] = cnn_metrics
models.append(('CNN', create_cnn_model))

for model_name, model_constructor in models:
    model = model_constructor() if callable(model_constructor) else model_constructor
    if model_name == 'CNN':
        X_train_reshaped = X_train.reshape(-1, 1, 1, 1)
        X_test_reshaped = X_test.reshape(-1, 1, 1, 1)
        threshold_metrics = evaluate_model_with_thresholds(model_name, model, X_train_reshaped, y_train, X_test_reshaped, y_test, thresholds)
    else:
        threshold_metrics = evaluate_model_with_thresholds(model_name, model, X_train, y_train, X_test, y_test, thresholds)
    model_threshold_metrics[model_name] = threshold_metrics

for model_name, metrics in model_threshold_metrics.items():
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, metrics['Accuracy'], label='Accuracy')
    plt.plot(thresholds, metrics['Recall'], label='Recall')
    plt.plot(thresholds, metrics['Specificity'], label='Specificity')

    plt.title(f'{model_name} Performance Over Different Thresholds')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Metric Score')
    plt.legend(frameon=False)
    plt.grid(True)
    plt.savefig(f'{model_name}_Performance_Thresholds.png', format='png', dpi=300)
    plt.show()

    if isinstance(metrics['AUC-ROC'], list):
        mean_auc_roc = sum(metrics['AUC-ROC']) / len(metrics['AUC-ROC'])
    else:
        mean_auc_roc = metrics['AUC-ROC']
    plt.figure(figsize=(12, 8))
    plt.plot(metrics['False Positive Rate'], metrics['Recall'], label=f'ROC Curve (area = {mean_auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Recall (True Positive Rate)')
    plt.legend(frameon=False)
    plt.grid(True)
    plt.savefig(f'{model_name}_ROC_Curve.png', format='png', dpi=300)
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(metrics['Recall'], metrics['Precision'], label='Precision-Recall Curve')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(frameon=False)
    plt.grid(True)
    plt.savefig(f'{model_name}_Precision_Recall.png', format='png', dpi=300)
    plt.show()
