import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np  # Import numpy to fix LabelEncoder issue

# Load the datasets
students_df = pd.read_csv(r"C:\Users\LENOVO\Downloads\university_recommendation_data_no_id (1).csv")
universities_df = pd.read_csv(r"C:\Users\LENOVO\Downloads\university_data.csv")

# Convert 'Country' columns to string type
students_df["Country"] = students_df["Country"].astype(str)
universities_df["Country"] = universities_df["Country"].astype(str)

# Encode 'Country' column using only known values from universities_df
le_country = LabelEncoder()
universities_df["Country"] = le_country.fit_transform(universities_df["Country"])

# Handle unseen countries in students_df
known_countries = le_country.classes_.tolist()  # Convert to list for easy checking
students_df["Country"] = students_df["Country"].apply(lambda x: x if x in known_countries else "Unknown")

# Fix: Ensure "Unknown" is added correctly to LabelEncoder
if "Unknown" not in known_countries:
    known_countries.append("Unknown")
le_country.classes_ = np.array(known_countries)  # Convert back to numpy array

# Transform the country names in students_df
students_df["Country"] = le_country.transform(students_df["Country"])

# Merge student data with universities based on country
merged_df = students_df.merge(universities_df, on="Country", how="inner")

# Selecting relevant features for university prediction
features = ["CGPA", "GRE_Score", "TOEFL_Score", "Work_Experience", "Research_Papers",
            "World_Rank", "Average_Tuition_Fees", "Acceptance_Rate", "Student_Faculty_Ratio", "Employability_Rating"]
X_university = merged_df[features]
y_university = merged_df["University_Name"]

# Splitting dataset into training and testing
X_university_train, X_university_test, y_university_train, y_university_test = train_test_split(
    X_university, y_university, test_size=0.2, random_state=42)

# Scaling numerical features
scaler = StandardScaler()
X_university_train = scaler.fit_transform(X_university_train)
X_university_test = scaler.transform(X_university_test)

# Train a Random Forest Classifier for university prediction
university_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
university_model.fit(X_university_train, y_university_train)

# Predictions
y_university_pred = university_model.predict(X_university_test)

# Evaluate the model
accuracy_university = accuracy_score(y_university_test, y_university_pred)
print(f"University Prediction Model Accuracy: {accuracy_university:.2f}")

# Encode University Names
le_university = LabelEncoder()
merged_df["University_Name"] = le_university.fit_transform(merged_df["University_Name"])

# Prepare data for country prediction
X_country = merged_df[["University_Name"]]
y_country = merged_df["Country"]

# Splitting dataset into training and testing
X_country_train, X_country_test, y_country_train, y_country_test = train_test_split(
    X_country, y_country, test_size=0.2, random_state=42)

# Train a Random Forest Classifier for country prediction
country_model = RandomForestClassifier(n_estimators=100, random_state=42)
country_model.fit(X_country_train, y_country_train)

# Predictions
y_country_pred = country_model.predict(X_country_test)

# Evaluate the model
accuracy_country = accuracy_score(y_country_test, y_country_pred)
print(f"Country Prediction Model Accuracy: {accuracy_country:.2f}")

def predict_country(university_name):
    """Predicts the country of a given university name.

    Args:
        university_name (str): The name of the university.

    Returns:
        str: The predicted country.
    """
    if university_name not in le_university.classes_:
        return "University not found in dataset"

    university_encoded = le_university.transform([university_name])[0]

    # Make the prediction
    predicted_country_encoded = country_model.predict([[university_encoded]])

    # Decode the country
    predicted_country = le_country.inverse_transform(predicted_country_encoded)[0]

    return predicted_country

def predict_university(student_profile):
    """Predicts the university based on student profile.
    
    Args:
        student_profile (dict): Dictionary containing student's details.
        
    Returns:
        str: Predicted university name.
    """
    # Create DataFrame from student profile
    student_df = pd.DataFrame([student_profile])

    # Ensure all columns are present
    for col in features:
        if col not in student_df.columns:
            student_df[col] = 0  # Use a more appropriate imputation if needed

    student_df = student_df[features]  # Ensure correct column order
    
    # Scale the data
    student_scaled = scaler.transform(student_df)
    
    # Predict the university
    predicted_university_encoded = university_model.predict(student_scaled)[0]
    
    return predicted_university_encoded

def get_student_input():
    """Gets student input from the user."""
    print("\nEnter student details for university prediction:")
    
    try:
        student_profile = {
            "CGPA": float(input("Enter CGPA (0-10 scale): ")),
            "GRE_Score": int(input("Enter GRE Score (260-340): ")),
            "TOEFL_Score": int(input("Enter TOEFL Score (0-120): ")),
            "Work_Experience": int(input("Enter Work Experience (years): ")),
            "Research_Papers": int(input("Enter number of Research Papers: ")),
            "World_Rank": int(input("Enter World Rank of University preference: ")),
            "Average_Tuition_Fees": float(input("Enter Average Tuition Fees (in USD): ")),
            "Acceptance_Rate": float(input("Enter Acceptance Rate (0-100%): ")),
            "Student_Faculty_Ratio": int(input("Enter Student Faculty Ratio: ")),
            "Employability_Rating": int(input("Enter Employability Rating (0-100): "))
        }
    except ValueError:
        print("Invalid input! Please enter numbers only.")
        return None
    
    return student_profile

# Get student input for university prediction
sample_student = get_student_input()
if sample_student:
    predicted_university = predict_university(sample_student)
    print(f"\nPredicted University: {predicted_university}")

    # Get the country of the predicted university
    predicted_country = predict_country(predicted_university)
    print(f"Predicted Country: {predicted_country}")
