# Mood Prediction Based on Fitness Data

## Project Overview
This project aims to predict a user's mood based on their fitness activity data, focusing on key variables such as the number of steps taken, distance covered, minutes of exercise, hours of sleep, average heart rate, and calories burned. By analyzing these physical activity metrics, we attempt to infer how the user's fitness routine correlates with their caloric expenditure, which may influence their overall mood.

The machine learning model is built using a neural network to predict **calories burned** based on the input fitness data. The primary goal of this model is to uncover the relationship between fitness metrics and caloric output, which in turn helps us predict mood based on physical activity.

## Dataset
The dataset used for this project contains various fitness-related attributes, including:
- `steps`: The number of steps the user took during a given period.
- `distance_km`: The total distance covered by the user in kilometers.
- `active_minutes`: The number of active minutes (exercise) recorded.
- `sleep_hours`: The number of hours the user slept.
- `heart_rate_avg`: The user's average heart rate.
- `calories_burned`: The number of calories the user burned.

These features are used to predict the number of calories burned, which serves as a proxy for estimating the user's mood based on physical activity.

## Project Workflow
1. **Data Preprocessing**: 
   - The dataset is loaded and cleaned (handling missing values, outliers, etc.).
   - Features are selected and normalized using `StandardScaler` to ensure all data is on the same scale.
   
2. **Model Training**:
   - A feed-forward neural network is created using TensorFlow/Keras.
   - The network consists of two hidden layers with `ReLU` activation functions and an output layer that predicts caloric expenditure.
   - The model is trained using the Adam optimizer and mean squared error (MSE) as the loss function.

3. **Model Evaluation**:
   - The model is evaluated on a test set, using metrics like **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)** to assess performance.
   - We visualize the training and validation loss, as well as the mean absolute error over the training epochs.

## Results
The neural network model is trained to predict the number of calories burned based on the provided fitness data. With enough training data, the model can help infer the user’s mood by analyzing how their physical activity affects their caloric output.

## Future Work
- Fine-tuning the model by experimenting with more advanced architectures (e.g., adding dropout layers or increasing the number of neurons).
- Collecting more data to improve accuracy and to allow for predicting mood directly.
- Exploring additional fitness and health metrics that may improve the quality of predictions, such as stress levels, dietary intake, or mood surveys.

## Requirements
- Python 3.x
- TensorFlow/Keras
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

To install the necessary libraries, you can run:
```bash
pip install -r requirements.txt
