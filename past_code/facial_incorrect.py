import glob
import os
import pandas as pd

folder_path = '../fyp data/facial'
all_files_facial = glob.glob(folder_path + '/*.csv')
label = pd.read_csv('../fyp data/metadata_mapped.csv')

relevant_au = [
    'AU01_c', 'AU04_c', 'AU06_c', 'AU12_c', 'AU15_c', 'AU17_c'
]

def extract_participant_id(file_name):
    return os.path.basename(file_name).split('_')[0]

def calculate_au_frequencies(files, labels, threshold=10):
    depressed_frequencies = []
    non_depressed_frequencies = []

    for file in files:
        participant_id = extract_participant_id(file)
        df = pd.read_csv(file)
        file_labels = labels[labels['Participant_ID'] == int(participant_id)]

        if len(file_labels) > 0:
            total_frequency = sum(df[au].sum() for au in relevant_au)
            if file_labels.iloc[0]['PHQ_Score'] >= threshold:
                depressed_frequencies.append(total_frequency)
            else:
                non_depressed_frequencies.append(total_frequency)

    avg_depressed_frequency = sum(depressed_frequencies) / len(depressed_frequencies) if depressed_frequencies else 0
    avg_non_depressed_frequency = sum(non_depressed_frequencies) / len(
        non_depressed_frequencies) if non_depressed_frequencies else 0

    return avg_depressed_frequency, avg_non_depressed_frequency

# Calculate average frequencies
avg_depressed_frequency, avg_non_depressed_frequency = calculate_au_frequencies(all_files_facial, label)

print(f"Average AU Frequency for Depressed Participants: {avg_depressed_frequency}")
print(f"Average AU Frequency for Non-Depressed Participants: {avg_non_depressed_frequency}")

correct_predictions = 0
total_samples = 0
detection_results = {}

# Iterate through each facial file to classify participants and compare with PHQ scores
for file in all_files_facial:
    participant_id = extract_participant_id(file)
    df = pd.read_csv(file)
    file_labels = label[label['Participant_ID'] == int(participant_id)]

    if len(file_labels) > 0:
        total_frequency = sum(df[au].sum() for au in relevant_au)
        average_frequency = total_frequency / len(df)
        depression_detected = average_frequency > avg_non_depressed_frequency
        actual_depression = file_labels.iloc[0]['PHQ_Score'] >= 10

        detection_results[participant_id] = {'Depression_Detected': depression_detected, 'Actual_Depression': actual_depression}

        if depression_detected == actual_depression:
            correct_predictions += 1
        total_samples += 1

# Calculate and print the accuracy
accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
print(f"Accuracy: {accuracy}%")

# Print out detailed detection results for each participant
for participant_id, results in detection_results.items():
    print(f"Participant ID: {participant_id}, Depression Detected: {results['Depression_Detected']}, Actual Depression: {results['Actual_Depression']}")
