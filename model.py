import numpy as np
import pandas as pd

# Load dataset
breast = pd.read_csv('dataset.csv')

# Basic info
print(breast.head())
print(breast.shape)
print("Duplicates:", breast.duplicated().sum())
print(breast.info())

# Drop unnecessary columns (if present)
if 'id' in breast.columns:
    breast.drop('id', axis=1, inplace=True)

if 'Unnamed: 32' in breast.columns:
    breast.drop('Unnamed: 32', axis=1, inplace=True)

print(breast.describe())
print(breast['diagnosis'].value_counts())

# Convert target to numeric
breast['diagnosis'] = breast['diagnosis'].map({"M": 1, "B": 0})
print(breast.corr())

# Split features and target
X = breast.drop('diagnosis', axis=1)
Y = breast['diagnosis']

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, Y_train)

# Predictions
y_pred = lg.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(Y_test, y_pred))

# Sample row
print("Sample scaled row:", X_train[10])

# ====== CUSTOM INPUT PREDICTION ======

inputvar = (
    -0.23711093,-0.4976419,0.61365274,-0.49813131,-0.53102815,-0.57694824,
    -0.17494424,-0.36215622,-0.284859,0.43345165,0.17818232,-0.36844966,
    0.55310406,-0.31671104,-0.40524636,0.04025752,-0.03795529,-0.18043065,
    0.16478901,-0.12170969,0.23079329,-0.50044002,0.81940367,-0.46922838,
    -0.53308833,-0.04910117,-0.04160193,-0.14913653,0.09681787,0.10617647,
    0.49035329
)
inputvar= X.iloc[0].values
# Convert + scale input
np_df = np.array(inputvar).reshape(1, -1)
np_df = sc.transform(np_df)

# Predict
predict = lg.predict(np_df)

# Output result
if predict[0] == 1:
    print("Cancerous")
else:
    print("Not Cancerous")
