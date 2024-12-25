# Data Processing Project with Streamlit

This application is a comprehensive data processing tool built using Streamlit, designed to guide users through the complete lifecycle of a data science project. Below is an overview of its key features:

## Key Features

1. **File Upload**:
   - Users can upload a CSV file, which will be read into a DataFrame for further processing.

2. **Data Preview**:
   - After uploading, users can preview the dataset, including options to view the head, tail, or a specified number of rows.

3. **Data Exploration**:
   - Users can view the entire dataset, column names, dimensions, and summary statistics.

4. **Data Cleaning**:
   - **Numeric Feature Conversion**: Users can check and convert numeric features.
   - **Handling Missing Values**: Users can identify missing values and choose strategies to handle them (mean, median, mode).
   - **Dropping Columns and Rows**: Users can drop specified columns or rows containing NaN values.

5. **Handling Duplicates**:
   - Users can check for and remove duplicate rows from the dataset.

6. **Categorical Data Handling**:
   - Users can apply one-hot encoding and label encoding to categorical features to prepare them for modeling.

7. **Column Management**:
   - Users can rename columns and change their data types, with options to confirm changes before applying.

8. **Model Training**:
   - The application allows users to select features and a target column for machine learning models. Users can choose algorithms like XGBoost, Support Vector Machine, or Random Forest to train models and assess their accuracy.

9. **Dataset Download**:
   - Users can download the cleaned and processed dataset as a CSV file.

10. **Error Handling**:
    - The app includes error handling to alert users if issues arise during processing.

## User Experience

The application is user-friendly, with a clear layout and interactive elements such as checkboxes, sliders, and buttons, making it accessible for both novice and experienced users in data science. The use of session state allows for persistent data management throughout the session.

Overall, this application provides a streamlined interface for data cleaning, exploration, and modeling, facilitating an efficient workflow for data scientists and analysts. 

## Installation

1. **Clone the repository**:
   Open your terminal or command prompt and run the following command:
   ```bash
   git clone https://github.com/mohamed202100/Streamlit_App.git

## 
2. **Navigate to the project directory**:
   Change into the directory of the cloned repository
   ```bash
   cd Streamlit_App

##
3. **Install the required libraries**:
   Use the following command to install all the necessary libraries specified in the requirements.txt file
   ```bash
   pip install -r requirements.txt

## 
4. **Run the application**:
   Start the Streamlit application with this command:
   ```bash
   streamlit run f.py

YouTube Video
Watch the tutorial on how to use the application: []
