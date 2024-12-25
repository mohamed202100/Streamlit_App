""" import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

st.title("Project name >>>>")
st.subheader("Complete Model Lifecycle")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload a CSV file")

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Load the data into session_state
        if "data" not in st.session_state:
            st.session_state.data = pd.read_csv(uploaded_file)

        data = st.session_state.data.copy()  # Work with a copy of the data for operations

        st.write("File successfully loaded!")
        st.dataframe(data)  # Display the dataframe

        # Preview dataset options
        if st.checkbox("Preview Dataset"):
            if st.button("Head"):
                st.write(data.head())
            elif st.button("Tail"):
                st.write(data.tail())
            else:
                number = st.slider("Select No of Rows", 1, data.shape[0])
                st.write(data.head(number))

        # Show entire data
        if st.checkbox("Show all data"):
            st.write(data)

        # Show column names
        if st.checkbox("Show Column Names"):
            st.write(data.columns)

        # Show dimensions
        if st.checkbox("Show Dimensions"):
            st.write(data.shape)

        # Show summary
        if st.checkbox("Show Summary"):
            st.write(data.describe())

        ##############################################################
        st.subheader("Start Cleaning")

        # Convert numeric features
        col_option = st.selectbox("Choose your option", ("Check numeric features", "Show unique values of categorical features"))
        numeric_columns = list(data.select_dtypes(include=np.number).columns)
        if col_option == "Check numeric features":
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            st.write("Done")
        elif col_option == "Show unique values of categorical features":
            for column in data.select_dtypes(include="object").columns:
                st.write(f"{column} : {data[column].unique()}")
                st.write("====================================")

        ###########################################################################
        st.subheader("Handle Missing Values")
        if st.checkbox("Show Missing Values"):
            st.write(data.isna().sum())

        # Treat missing values
        col_option = st.selectbox("Select column to treat missing values", data.columns)
        missing_values_clear = st.selectbox("Select Missing values Strategy", ("Replace with Mode", "Replace with Mean", "Replace with Median"))

        if missing_values_clear == "Replace with Mean":
            replaced_value = data[col_option].mean()
        elif missing_values_clear == "Replace with Median":
            replaced_value = data[col_option].median()
        elif missing_values_clear == "Replace with Mode":
            replaced_value = data[col_option].mode()[0]

        Replace = st.selectbox("Replace values of column?", ("Yes", "No"))
        if Replace == "Yes":
            data[col_option] = data[col_option].fillna(replaced_value)
            st.success("Null values replaced!")

        ##########################################################################
        st.subheader("Drop Columns and Rows")

        # Drop chosen columns
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns)

        if st.button("Drop Columns"):
            if columns_to_drop:
                data = data.drop(columns=columns_to_drop, axis=1)
                st.session_state.data = data  # Save changes to session_state
                st.success(f"Columns {columns_to_drop} have been dropped!")
            else:
                st.warning("Please select at least one column to drop.")

        # Drop rows with NaN values
        columns_to_check = st.multiselect("Select columns to check for NaN values:", data.columns)

        if st.button("Drop Rows with NaN"):
            if columns_to_check:
                data = data.dropna(subset=columns_to_check)
                st.session_state.data = data  # Save changes to session_state
                st.success(f"Rows with NaN values in columns {columns_to_check} have been dropped!")
            else:
                st.warning("Please select at least one column to check for NaN values.")

        st.dataframe(data)

        ##########################################################################
        st.subheader("Handle Categorical Values")
        categorical_cols_features = list(data.select_dtypes(include="object").columns)

        # One-hot encoding
        one_hot_enc = st.multiselect("Select nominal categorical columns", data.columns)
        for column in one_hot_enc:
            if data[column].dtype == 'object':  # Apply to categorical/string columns
                data = pd.get_dummies(data, columns=[column])

        # Label encoding
        label_encoder = LabelEncoder()
        label_enc = st.multiselect("Select ordinal categorical columns", data.columns)
        for column in label_enc:
            if data[column].dtype == 'object':
                data[column] = label_encoder.fit_transform(data[column])

        st.dataframe(data)

        ############################################################################################
        # Change column names
        st.subheader("Change Column Names")
        rename_column = st.selectbox("Select column to rename", data.columns)
        new_column_name = st.text_input(f"Enter new name for {rename_column}")

        if st.button("Rename Column"):
            if new_column_name:
                data.rename(columns={rename_column: new_column_name}, inplace=True)
                st.session_state.data = data  # Save changes to session_state
                st.success(f"Column {rename_column} has been renamed to {new_column_name}")
            else:
                st.warning("Please enter a new column name.")

        st.dataframe(data)

        ##########################################################################
        # Change Data Type of Column
        st.subheader("Change Data Type of Column")
        dtype_column = st.selectbox("Select column to change its data type", data.columns)
        new_dtype = st.selectbox("Select new data type", ("int64", "float64", "object"))

        if st.button("Change Data Type"):
            try:
                data[dtype_column] = data[dtype_column].astype(new_dtype)
                st.session_state.data = data  # Save changes to session_state
                st.success(f"Data type of column {dtype_column} has been changed to {new_dtype}")
            except Exception as e:
                st.error(f"Error changing data type: {e}")

        st.dataframe(data)

        ##########################################################################
        st.title("Machine Learning models")
        

        # Select features and labels
        features = st.multiselect("Select Feature Columns", data.columns)
        labels = st.multiselect("Select Target Column", data.columns)

        if features and labels:
            X = data[features].values
            y = data[labels].values

            train_percent = st.slider("Select % to train model", 1, 100) / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percent, random_state=1)

            alg = ['XGBoost Classifier', 'Support Vector Machine', 'Random Forest Classifier']
            classifier = st.selectbox('Which algorithm?', alg)
            if classifier == 'XGBoost Classifier':
                XG = XGBClassifier()
                XG.fit(X_train, y_train)
                st.write('Accuracy:', XG.score(X_test, y_test))
            elif classifier == 'Support Vector Machine':
                svm = SVC()
                svm.fit(X_train, y_train)
                st.write('Accuracy:', svm.score(X_test, y_test))
            elif classifier == 'Random Forest Classifier':
                RFC = RandomForestClassifier()
                RFC.fit(X_train, y_train)
                st.write('Accuracy:', RFC.score(X_test, y_test))

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a CSV file to proceed.")
 """


import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import io

st.title("Project Name >>>>")
st.subheader("Complete Model Lifecycle")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload a CSV file")

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Load the data into session_state
        if "data" not in st.session_state:
            st.session_state.data = pd.read_csv(uploaded_file)

        data = st.session_state.data.copy()  # Work with a copy of the data for operations

        st.write("File successfully loaded!")
        st.dataframe(data)  # Display the dataframe

        ##############################################################
        st.subheader("Start Cleaning")

        # Convert numeric features
        col_option = st.selectbox("Choose your option", ("Check numeric features", "Show unique values of categorical features"))
        numeric_columns = list(data.select_dtypes(include=np.number).columns)
        if col_option == "Check numeric features":
            data_before = data.copy()
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            st.write("Before:")
            st.dataframe(data_before)
            st.write("After:")
            st.dataframe(data)
            if st.button("Confirm Numeric Conversion"):
                st.success("Numeric conversion applied!")
        elif col_option == "Show unique values of categorical features":
            for column in data.select_dtypes(include="object").columns:
                st.write(f"{column} : {data[column].unique()}")
                st.write("====================================")

        ###########################################################################
        st.subheader("Handle Missing Values")
        if st.checkbox("Show Missing Values"):
            st.write(data.isna().sum())

        # Treat missing values
        col_option = st.selectbox("Select column to treat missing values", data.columns)
        missing_values_clear = st.selectbox("Select Missing values Strategy", ("Replace with Mode", "Replace with Mean", "Replace with Median"))

        if missing_values_clear == "Replace with Mean":
            replaced_value = data[col_option].mean()
        elif missing_values_clear == "Replace with Median":
            replaced_value = data[col_option].median()
        elif missing_values_clear == "Replace with Mode":
            replaced_value = data[col_option].mode()[0]

        Replace = st.selectbox("Replace values of column?", ("Yes", "No"))
        if Replace == "Yes":
            data_before = data.copy()
            data[col_option] = data[col_option].fillna(replaced_value)
            st.success("Null values replaced!")
            st.write("Before:")
            st.dataframe(data_before)
            st.write("After:")
            st.dataframe(data)
            if st.button("Confirm Missing Value Replacement"):
                st.success("Missing values replacement confirmed!")

        ##########################################################################
        st.subheader("Drop Columns and Rows")

        # Drop chosen columns
        columns_to_drop = st.multiselect("Select columns to drop:", data.columns)

        if st.button("Drop Columns"):
            if columns_to_drop:
                data_before = data.copy()
                data = data.drop(columns=columns_to_drop, axis=1)
                st.session_state.data = data  # Save changes to session_state
                st.success(f"Columns {columns_to_drop} have been dropped!")
                st.write("Before:")
                st.dataframe(data_before)
                st.write("After:")
                st.dataframe(data)
                if st.button("Confirm Column Drop"):
                    st.success("Columns dropped confirmed!")
            else:
                st.warning("Please select at least one column to drop.")

        # Drop rows with NaN values
        columns_to_check = st.multiselect("Select columns to check for NaN values:", data.columns)

        if st.button("Drop Rows with NaN"):
            if columns_to_check:
                data_before = data.copy()
                data = data.dropna(subset=columns_to_check)
                st.session_state.data = data  # Save changes to session_state
                st.success(f"Rows with NaN values in columns {columns_to_check} have been dropped!")
                st.write("Before:")
                st.dataframe(data_before)
                st.write("After:")
                st.dataframe(data)
                if st.button("Confirm Row Drop"):
                    st.success("Rows with NaN values dropped confirmed!")
            else:
                st.warning("Please select at least one column to check for NaN values.")

        ##########################################################################
        st.subheader("Handle Duplicates")

        # Check for duplicates
        if st.checkbox("Show Duplicates"):
            st.write(data.duplicated().sum())

        # Drop duplicates
        if st.button("Drop Duplicates"):
            data_before = data.copy()
            data = data.drop_duplicates()
            st.session_state.data = data  # Save changes to session_state
            st.success("Duplicate rows have been dropped!")
            st.write("Before:")
            st.dataframe(data_before)
            st.write("After:")
            st.dataframe(data)
            if st.button("Confirm Duplicate Removal"):
                st.success("Duplicate rows removal confirmed!")

        ##########################################################################
        st.subheader("Handle Categorical Values")
        categorical_cols_features = list(data.select_dtypes(include="object").columns)

        # One-hot encoding
        one_hot_enc = st.multiselect("Select nominal categorical columns", data.columns)
        for column in one_hot_enc:
            if data[column].dtype == 'object':  # Apply to categorical/string columns
                data_before = data.copy()
                data = pd.get_dummies(data, columns=[column])
                st.session_state.data = data  # Save changes to session_state
                st.write("Before:")
                st.dataframe(data_before)
                st.write("After:")
                st.dataframe(data)
                if st.button("Confirm One-Hot Encoding"):
                    st.success("One-Hot Encoding confirmed!")

        # Label encoding
        label_encoder = LabelEncoder()
        label_enc = st.multiselect("Select ordinal categorical columns", data.columns)
        for column in label_enc:
            if data[column].dtype == 'object':
                data_before = data.copy()
                data[column] = label_encoder.fit_transform(data[column])
                st.session_state.data = data  # Save changes to session_state
                st.write("Before:")
                st.dataframe(data_before)
                st.write("After:")
                st.dataframe(data)
                if st.button("Confirm Label Encoding"):
                    st.success("Label Encoding confirmed!")

        ##########################################################################
        # Change column names
        st.subheader("Change Column Names")
        rename_column = st.selectbox("Select column to rename", data.columns)
        new_column_name = st.text_input(f"Enter new name for {rename_column}")

        if st.button("Rename Column"):
            if new_column_name:
                data_before = data.copy()
                data.rename(columns={rename_column: new_column_name}, inplace=True)
                st.session_state.data = data  # Save changes to session_state
                st.success(f"Column {rename_column} has been renamed to {new_column_name}")
                st.write("Before:")
                st.dataframe(data_before)
                st.write("After:")
                st.dataframe(data)
                if st.button("Confirm Column Rename"):
                    st.success("Column rename confirmed!")

        ##########################################################################
        # Download Button
        st.subheader("Download the Dataset")
        
        # Convert the dataframe to CSV and enable the user to download it
        @st.cache
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(data)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='cleaned_data.csv',
            mime='text/csv'
        )

        st.dataframe(data)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a CSV file to proceed.")
