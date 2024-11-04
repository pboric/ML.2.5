import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline as sk_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import joblib  # Import joblib for saving the model

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

data = pd.read_csv(r"C:\Users\kresi\OneDrive\Desktop\Turing college\Project9\healthcare-dataset-stroke-data.csv")
data['bmi'].fillna(data['bmi'].mean(), inplace=True)

le = LabelEncoder()
text_data_features = [
    'gender', 'ever_married', 'work_type', 
    'Residence_type', 'smoking_status'
]
for feature in text_data_features:
    data[feature] = le.fit_transform(data[feature])

over = SMOTE(sampling_strategy=1)
under = RandomUnderSampler(sampling_strategy=0.1)
f1 = data.loc[:, :'smoking_status']
t1 = data.loc[:, 'stroke']
steps = [('under', under), ('over', over)]
pipeline = Pipeline(steps=steps)
f1, t1 = pipeline.fit_resample(f1, t1)
Counter(t1)

x_train, x_test, y_train, y_test = train_test_split(
    f1, t1, test_size=0.15, random_state=2
)

def select_features(x_train, y_train, x_test):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(x_train, y_train)
    x_train_fs = fs.transform(x_train)
    x_test_fs = fs.transform(x_test)
    return x_train_fs, x_test_fs, fs

x_train_fs, x_test_fs, fs = select_features(x_train, y_train, x_test)

pipeline_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=mutual_info_classif, k='all')),
    ('classifier', XGBClassifier())
])

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=mutual_info_classif, k='all')),
    ('classifier', RandomForestClassifier())
])

param_grid_xgb = {
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__n_estimators': [100, 500, 1000]
}

param_grid_rf = {
    'classifier__n_estimators': [100, 500, 1000],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search_xgb = GridSearchCV(
    pipeline_xgb, param_grid_xgb, cv=5, scoring='roc_auc'
)
grid_search_rf = GridSearchCV(
    pipeline_rf, param_grid_rf, cv=5, scoring='roc_auc'
)

grid_search_xgb.fit(x_train_fs, y_train)
grid_search_rf.fit(x_train_fs, y_train)

ensemble = VotingClassifier(estimators=[
    ('xgb', grid_search_xgb.best_estimator_),
    ('rf', grid_search_rf.best_estimator_)
], voting='soft')

ensemble.fit(x_train_fs, y_train)

def model_evaluation(model, x_test, y_test):
    prediction = model.predict(x_test)
    print("Classification Report:\n", classification_report(y_test, prediction))
    print("ROC_AUC Score: ", '{0:.2%}'.format(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])))

model_evaluation(ensemble, x_test_fs, y_test)

joblib.dump(ensemble, 'ensemble_model.pkl')
print("Model saved as 'ensemble_model.pkl'")
