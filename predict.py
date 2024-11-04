import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

new_data = pd.read_csv("C:\\Users\kresi\\OneDrive\\Desktop\\Turing college\\Project9\\healthcare-dataset-stroke-data.csv")

new_data = new_data.drop('id', axis=1)

text_data_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

le = LabelEncoder()
scaler = StandardScaler()

for feature in text_data_features:
    new_data[feature] = le.fit_transform(new_data[feature])

new_data['bmi'].fillna(new_data['bmi'].mean(), inplace=True)

numerical_features = ['age', 'bmi', 'avg_glucose_level']

scaler.fit(new_data[numerical_features])

new_data[numerical_features] = scaler.transform(new_data[numerical_features])

model = joblib.load('ensemble_model.pkl')

predictions = model.predict(new_data)

print(predictions)
