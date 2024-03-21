import pandas as pd
import glob
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load PHQ scores
phq_scores = pd.read_csv('../fyp data/metadata_mapped.csv')  # Update this path

# Correct Participant_ID in phq_scores to int if it's not already
phq_scores['Participant_ID'] = phq_scores['Participant_ID'].astype(str).str.extract('(\d+)').astype(int)

# Get the list of text data files
folder_path = '../fyp data/text'  # Update this path
all_files = glob.glob(os.path.join(folder_path, '*.csv'))


# Function to extract participant ID from file name
def extract_participant_id(file_name):
    return int(re.match(r'(\d+)_', os.path.basename(file_name)).group(1))


# Function to preprocess text
def preprocess_text(text, lemmatizer, stop_words):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize an empty list to store participant data with processed text
participant_text_data = []

# Loop through files, read data, extract participant ID
for file in all_files:
    participant_id = extract_participant_id(file)
    temp_df = pd.read_csv(file)

    # Concatenate all the text data into one string for each participant
    all_text = ' '.join(temp_df.fillna('').astype(str).sum(axis=0).tolist())

    # Preprocess the text data
    processed_text = preprocess_text(all_text, lemmatizer, stop_words)

    participant_text_data.append({
        'Participant_ID': participant_id,
        'Processed_Text': processed_text
    })

# Create a DataFrame from the list of participant text data
text_data = pd.DataFrame(participant_text_data)

# Merge the text data with PHQ scores on Participant_ID
merged_data = text_data.merge(phq_scores, on='Participant_ID')

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(merged_data['Processed_Text']).toarray()

# Labels for classification based on PHQ score
y = merged_data['PHQ_Score'].apply(lambda x: x >= 10).values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Printing the actual PHQ-based labels and predictions for
# Add the PHQ label to the DataFrame
merged_data['PHQ_Label'] = merged_data['PHQ_Score'].apply(lambda x: True if x >= 10 else False)

# Predict for the entire dataset to label each participant
full_predictions = model.predict(X)
merged_data['Prediction'] = full_predictions

# Print out the depression status for each participant as True or False based on PHQ score
for index, row in merged_data.iterrows():
    print(f"Participant ID: {row['Participant_ID']}, PHQ Label: {row['PHQ_Label']}, Prediction: {bool(row['Prediction'])}")

# Calculate the accuracy based on the full dataset
full_accuracy = accuracy_score(merged_data['PHQ_Label'], merged_data['Prediction'])
print("Full Model Accuracy:", full_accuracy * 100)

confusion_mat = confusion_matrix(merged_data['PHQ_Label'], merged_data['Prediction'])
print("Confusion Matrix:")
print(confusion_mat)