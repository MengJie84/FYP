import glob
import os
import pandas as pd

folder_path = '../fyp data/facial'
all_files_facial = glob.glob(folder_path + '/*.csv')
label = pd.read_csv('../fyp data/metadata_mapped.csv')

relevant_au = [
    'AU01_c', 'AU04_c', 'AU06_c', 'AU12_c', 'AU15_c', 'AU17_c'
]

# Check if any files are found
if not all_files_facial:
    print("No files found in the specified folder.")
else:
    print(f"Found {len(all_files_facial)} files.")

# Initialize a list to store the results
results = []

def extract_participant_id(file_name):
    return os.path.basename(file_name).split('_')[0]

# Iterate through each facial file
for file in all_files_facial:
    participant_id = extract_participant_id(file)

    # Load facial data
    df = pd.read_csv(file)
    facial_data = df[relevant_au]

    # Filter label data based on Participant ID
    file_labels = label[label['Participant_ID'] == int(participant_id)]

    # Calculate the total frequency of 1's in all relevant AU columns
    total_frequency = sum(df[au].sum() for au in relevant_au)

    # Determine depression status based on the total frequency
    depression_detected = total_frequency > 2.25

    # Check if the label indicates depression
    actual_depression = len(file_labels) > 0 and file_labels.iloc[0]['PHQ_Score'] >= 10

    # Append results to the list
    results.append({
        'Participant_ID': participant_id,
        'Depression_Label': actual_depression,
        'Depression_Detected': depression_detected
    })

# Print results
for result in results:
    print(result)

# Calculate accuracy
correct_predictions = sum(1 for res in results if res['Depression_Label'] == res['Depression_Detected'])
total_samples = len(results)
accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

# Print the accuracy
print(f"Accuracy: {accuracy}%")
