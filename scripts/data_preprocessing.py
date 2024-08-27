
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(df):
    # Example: Encoding categorical variables, handling missing values, etc.
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Feature scaling (optional, depending on the model)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_encoded)
    df_encoded = pd.DataFrame(scaled_features, columns=df_encoded.columns)
    
    return df_encoded

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    train_df, test_df = load_data(r'D:\Projects\MLflow project\data\train.csv', r'D:\Projects\MLflow project\data\test.csv')
    train_df_encoded = preprocess_data(train_df)
    test_df_encoded = preprocess_data(test_df)
    
    X_train, X_val, y_train, y_val = split_data(train_df_encoded, 'y_yes')
    X_test = test_df_encoded.drop(columns=['y_yes'])
    y_test = test_df_encoded['y_yes']
    
    # Optionally, save preprocessed data
    X_train.to_csv('X_train.csv', index=False)
    X_val.to_csv('X_val.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_val.to_csv('y_val.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
