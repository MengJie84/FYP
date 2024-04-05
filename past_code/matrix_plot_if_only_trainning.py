def plot_matrix(y, title):
    class_distribution = Counter(y)
    sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()))
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Class')
    plt.show()

X = df[['Prediction']].values
y = df['Actual'].values
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the training data
random_under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = random_under_sampler.fit_resample(X_train, y_train)

# Create a DataFrame from the resampled training data for visualization
df_train_resampled = pd.DataFrame(X_train_resampled, columns=['Prediction'])
df_train_resampled['Actual'] = y_train_resampled

# Plot the class distribution before and after balancing
print("Class distribution before balancing:")
plot_matrix(y_train, 'Class Distribution Before Balancing')
print("Class distribution after balancing:")
plot_matrix(y_train_resampled, 'Class Distribution After Balancing')

# Now you can use the balanced training data (X_train_resampled, y_train_resampled) for model training.
