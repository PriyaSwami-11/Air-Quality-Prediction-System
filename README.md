# Air-Quality-Prediction-System
AI-based air quality prediction using PM2.5, PM10, NO2, CO, temperature, and humidity. A TensorFlow neural network is trained on synthetic data and converted to TensorFlow Lite for efficient deployment on edge and IoT devices.
import pickle, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data: PM2.5, PM10, NO2, CO, Temperature, Humidity
np.random.seed(42)
X = np.random.uniform(
    [5,10,10,0.1,15,30],
    [150,300,80,5,40,90],
    (10000,6)
)

y = (0.5*X[:,0] + 0.25*X[:,1] + 0.3*X[:,2] +
     10*X[:,3] + np.random.normal(0,5,10000))

# Split and scale data
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),
    tf.keras.layers.Dense(32,activation="relu"),
    tf.keras.layers.Dense(16,activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam",loss="mse",metrics=["mae"])

# Train
model.fit(X_train,y_train,epochs=20,batch_size=32,verbose=0)

# Test
loss,mae = model.evaluate(X_test,y_test,verbose=0)
print("Test MAE:",mae)

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("aqi_model.tflite","wb") as f:
    f.write(tflite_model)

print("Model saved successfully!")
