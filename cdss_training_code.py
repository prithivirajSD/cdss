import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from disease_info import get_disease_info
import os

CDSS = pd.read_csv("data/symbipredict_2022.csv")
print(CDSS.head())

"""#III. Data Preparation

##1. Data Cleansing
"""

print(CDSS.shape)

print(CDSS.info())

features = CDSS.columns

features.to_series().to_csv('data/features.csv', index=False)

pd.Series(np.unique(CDSS[features[-1]])).to_csv('data/classes.csv', index=False)

"""###Handling Missing Values

 Let us check the total number of missing values in each column.
"""

print(CDSS.isnull().sum())

"""There is no missing values in the dataset. So, we can move onto the next step of data science process model.

#IV. Data Exploration
"""

CDSS = CDSS.drop_duplicates()
print(CDSS)

print(CDSS.describe())

"""###Visualizations
Using visualizations to understand the distribution of features and relationships between them by creating various techniques like scatter plots, count plots, etc.
"""

plt.figure(figsize=(10, 6))
sns.countplot(x='prognosis', data=CDSS)

plt.xticks(rotation=90)
plt.title("Count Plot of Diseases (Prognosis)")
plt.xlabel("Disease (Prognosis)")
plt.ylabel("Count")
plt.show()

symptom_sums = CDSS.groupby('prognosis').sum().reset_index()
symptom_sums_melted = symptom_sums.melt(id_vars='prognosis', var_name='Symptom', value_name='Count')

plt.figure(figsize=(12, 6))
sns.barplot(x='prognosis', y='Count', data=symptom_sums_melted)
plt.xticks(rotation=90)
plt.title('Sum of Symptoms per Disease')
plt.xlabel('Disease')
plt.ylabel('Sum of Symptoms')
plt.show()

"""#V. Data Modeling

##1. Model and Variable Selection

Let `X` be the new dataframe which comprises of independent variables and `y` be the dependent or target variable of our `CDSS`.
"""

X = CDSS.drop('prognosis', axis=1)
y = CDSS['prognosis']

"""**Let** us split the dataset into Train and Test sets in 80 : 20 ratio."""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Ensure these are pandas DataFrames/Series:
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

"""# RNN Bidirectional LSTM"""

X_train_reshaped = X_train.to_numpy().reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_reshaped = X_test.to_numpy().reshape((X_test.shape[0], 1, X_test.shape[1]))
y_train_array = y_train.to_numpy().reshape(-1, 1)
y_test_array = y_test.to_numpy().reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train_array)
y_test_encoded = encoder.transform(y_test_array)

# If y_test_encoded is a sparse matrix, convert to dense:
if hasattr(y_test_encoded, "toarray"):
    y_test_encoded_dense = y_test_encoded.toarray()
else:
    y_test_encoded_dense = y_test_encoded

y_test_encoded_dense = np.asarray(y_test_encoded_dense)

rnn_model = Sequential()

rnn_model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(1, X_train.shape[1])))
rnn_model.add(Dropout(0.2))

rnn_model.add(LSTM(64, return_sequences=False))
rnn_model.add(Dropout(0.2))

rnn_model.add(Dense(128, activation='relu'))
rnn_model.add(Dropout(0.2))

rnn_model.add(Dense(y_train_encoded.shape[1], activation='softmax'))

rnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = rnn_model.fit(X_train_reshaped, y_train_encoded, epochs=250, batch_size=64, validation_data=(X_test_reshaped, y_test_encoded))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

y_pred_rnn = rnn_model.predict(X_test_reshaped)
y_pred_rnn_class = np.argmax(y_pred_rnn, axis=1)
cm_rnn = confusion_matrix(np.argmax(y_test_encoded_dense, axis=1), y_pred_rnn_class)
sns.heatmap(cm_rnn, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.show()

print(classification_report(np.argmax(y_test_encoded_dense, axis=1), y_pred_rnn_class))

# Ensure model directory exists
os.makedirs("model", exist_ok=True)
rnn_model.save("model/CDSS_RNN.h5")


# Generate a sample dataframe with 10 samples (rows) and 132 binary features (columns)
num_samples = 10
num_features = len(features) - 1
sample_data = np.random.randint(0, 2, size=(num_samples, num_features))

# Create a DataFrame with the generated data
new_data = pd.DataFrame(sample_data, columns=features[:-1])

print(new_data)

# Reshape new data to match the input shape expected by the RNN model
new_data_reshaped = new_data.values.reshape((new_data.shape[0], 1, new_data.shape[1]))

# Predict probabilities for the new data
new_data_pred_probs = rnn_model.predict(new_data_reshaped)

# Convert probabilities to class labels (indices)
new_data_pred_classes = np.argmax(new_data_pred_probs, axis=1)

# Convert predicted class indices to original class labels using the encoder
predicted_labels = encoder.inverse_transform(np.eye(y_train_encoded.shape[1])[new_data_pred_classes])

# Output predictions
print("Predicted classes for new data:", predicted_labels)

print("Predicted disease:", predicted_labels[0])
info = get_disease_info(predicted_labels[0])
print("Disease info found:", info)

def display_disease_info(prediction):
    info = get_disease_info(prediction)
    st.title(f"{prediction} - Disease Overview")
    if info is None:
        st.warning("No information available for this disease.")
        return
    tabs = st.tabs(["Overview", "Tests & Treatment", "Safety Measures", "Specialist Contact Details"])
    with tabs[0]:
        # Show description if available
        description = info.get('description', None)
        if description:
            st.markdown(f"**Description:** {description}")

        # Show symptoms
        st.subheader("Common Symptoms")
        symptoms = info.get('Symptoms', [])
        if symptoms:
            for item in symptoms:
                st.markdown(f"- {item}")
        else:
            st.warning("No symptoms available.")

        # Show consulting specialist
        specialist = info.get('Consulting Specialist', None)
        if specialist:
            st.subheader("Consulting Specialist")
            if isinstance(specialist, list):
                for s in specialist:
                    st.markdown(f"- {s}")
            else:
                st.markdown(f"- {specialist}")

        # Show resource link
        resource = info.get('Resource Link', None)
        if resource:
            st.markdown(f"[More Information]({resource})")
    # ... rest of your code for other tabs ...