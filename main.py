from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


data_path = Path.cwd() / "data"
test_data_path = data_path / "test.csv"
train_data_path = data_path / "train.csv"
gender_submission_path = data_path / "gender_submission.csv"

test_data = pd.read_csv(test_data_path)
train_data = pd.read_csv(train_data_path)
gender_submission_data = pd.read_csv(gender_submission_path)

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": predictions})
output.to_csv("./out/submission.csv", index=False)
print("Submission completed...")
