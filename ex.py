import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:/year-2/ML/Placement_Prediction/Data/Raw Data/placement_predict_50k Dataset (2).csv")

print(df.head(10))
print(df.select_dtypes(include=['object', 'category']))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 1. Load the cleaned PlacementPredict dataset
# ---------------------------------------------------------

file_path = "data/processed/cleaned_data.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)


# ---------------------------------------------------------
# 2. Select numerical features
# ---------------------------------------------------------

numeric_features = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular"
]


# ---------------------------------------------------------
# 3. Separate features (X) and target (y)
# ---------------------------------------------------------

X = df[numeric_features]

y = df["PlacementStatus"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ---------------------------------------------------------
# 4. Split data into training and testing sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ---------------------------------------------------------
# 5. Create MinMaxScaler
# ---------------------------------------------------------

scaler = MinMaxScaler()


# ---------------------------------------------------------
# 6. Fit and transform ONLY the training data
# ---------------------------------------------------------

X_train_scaled = scaler.fit_transform(X_train)


# ---------------------------------------------------------
# 7. Transform the test data
# ---------------------------------------------------------

X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 8. Convert scaled arrays back to DataFrames
# ---------------------------------------------------------

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=numeric_features,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=numeric_features,
    index=X_test.index
)


# ---------------------------------------------------------
# 9. Display the scaled training data
# ---------------------------------------------------------

print("\nScaled Training Data:")
print(X_train_scaled.head())


# ---------------------------------------------------------
# 10. Display the scaled test data
# ---------------------------------------------------------

print("\nScaled Test Data:")
print(X_test_scaled.head())


# ---------------------------------------------------------
# 11. Check minimum and maximum values
# ---------------------------------------------------------

print("\nTraining Data Minimum:")
print(X_train_scaled.min().round(2))

print("\nTraining Data Maximum:")
print(X_train_scaled.max().round(2))


# ---------------------------------------------------------
# 12. Compare original and scaled CGPA
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Original_CGPA": X_train["CGPA"].head(10).values,
    "Scaled_CGPA": X_train_scaled["CGPA"].head(10).values
})

print("\nCGPA Before and After Min-Max Scaling:")
print(comparison)