import pandas as pd
import numpy as np
import joblib  # For saving the scaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras

# Load Data
df_train = pd.read_csv('exoTrain.csv')
df_test = pd.read_csv('exoTest.csv')
df = pd.concat([df_train, df_test], ignore_index=True)

# Clean Data
df.replace('-', np.nan, inplace=True)
df.fillna(0, inplace=True)

# Split X and y
X = df.drop('LABEL', axis=1)
y = df['LABEL'].replace({1: 0, 2: 1})
X = X.astype(np.float64)

# Split Train/Test
X_train = X.iloc[:len(df_train)]
y_train = y.iloc[:len(df_train)]
X_test = X.iloc[len(df_train):]
y_test = y.iloc[len(df_train):]

# Scale Data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the Scaler (CRITICAL FOR BACKEND)
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved to scaler.pkl")

# SMOTE (Fix Imbalance)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Reshape for CNN
X_train_reshaped = X_train_resampled.reshape(X_train_resampled.shape[0], X_train_resampled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Build Model
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_reshaped.shape[1], 1)),
    keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(64, 5, activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    X_train_reshaped,
    y_train_resampled,
    epochs=15, # Reduced for speed, increase if needed
    batch_size=64,
    validation_data=(X_test_reshaped, y_test),
    verbose=1
)

# Save Model (CRITICAL FOR BACKEND)
model.save('exoplanet_cnn_model.h5')
print("✅ Model saved to exoplanet_cnn_model.h5")