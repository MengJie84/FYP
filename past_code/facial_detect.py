import glob
import os
import pandas as pd

folder_path_facial = '../fyp data/facial'
all_files_facial = glob.glob(folder_path_facial + '/*.csv')
phq_scores = pd.read_csv('../fyp data/metadata_mapped.csv')

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
avg_depressed_frequency, avg_non_depressed_frequency = calculate_au_frequencies(all_files_facial, phq_scores)

print(f"Average AU Frequency for Depressed Participants: {avg_depressed_frequency}")
print(f"Average AU Frequency for Non-Depressed Participants: {avg_non_depressed_frequency}")

correct_predictions = 0
total_samples = 0

# Assuming the calculation of avg_non_depressed_frequency is done as before

# Placeholder for results mapping
depression_detection_results = {}

for file in all_files_facial:
    participant_id = extract_participant_id(file)
    df = pd.read_csv(file)
    file_labels = phq_scores[phq_scores['Participant_ID'] == int(participant_id)]

    if len(file_labels) > 0:
        # Calculate total frequency for the current file
        total_frequency = sum(df[au].sum() for au in relevant_au)

        # Determine if the total frequency indicates depression
        depression_detected = total_frequency > avg_non_depressed_frequency

        # Map the detection result with participant ID
        depression_detection_results[participant_id] = depression_detected

        # Get actual depression status based on PHQ_Score
        actual_depression = file_labels.iloc[0]['PHQ_Score'] >= 10

        # Print both detected and actual depression status
        print(
            f"Participant ID: {participant_id}, Detected Depression: {depression_detected}, Actual Depression: {actual_depression}")

# Optionally, if you want to calculate and print accuracy based on this approach
correct_predictions = sum(
    depression_detection_results[pid] == (phq_scores[phq_scores['Participant_ID'] == int(pid)].iloc[0]['PHQ_Score'] >= 10) for pid
    in depression_detection_results)
total_participants = len(depression_detection_results)

accuracy = (correct_predictions / total_participants) * 100 if total_participants > 0 else 0
print(f"Accuracy: {accuracy}%")
