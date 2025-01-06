import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import os

# Preprocessing functions
class DataPreprocessor:
    @staticmethod
    def preprocess_data(df):
        # Convert datetime columns to datetime objects
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

        # Feature engineering
        df['Weekday'] = df['Date'].dt.weekday
        df['IsWeekend'] = df['Weekday'].isin([5, 6]).astype(int)
        df['DaysToNextHoliday'] = DataPreprocessor.calculate_days_to_next_holiday(df['Date'])
        df['DaysSinceLastHoliday'] = DataPreprocessor.calculate_days_since_last_holiday(df['Date'])
        df['MonthPart'] = df['Day'].apply(lambda x: 'Begin' if x <= 10 else ('Mid' if x <= 20 else 'End'))

        # Handling NaN values
        df.fillna(-1, inplace=True)  # Replace missing values with -1

        # Drop columns that won't be used for modeling
        df.drop(['Year', 'Month', 'Day', 'Date'], axis=1, inplace=True)

        return df

    @staticmethod
    def calculate_days_to_next_holiday(dates):
        # Placeholder: Use real holiday data
        holidays = pd.to_datetime(['2013-01-01', '2015-07-31'])
        return dates.apply(lambda x: (holidays - x).min().days if x < holidays.max() else -1)

    @staticmethod
    def calculate_days_since_last_holiday(dates):
        # Placeholder: Use real holiday data
        holidays = pd.to_datetime(['2013-01-01', '2015-07-31'])
        return dates.apply(lambda x: (x - holidays[holidays <= x].max()).days if x > holidays.min() else -1)

# Model building functions
class ModelBuilder:
    @staticmethod
    def build_pipeline():
        numeric_features = ['Customers', 'CompetitionDistance', 'CompetitionOpen', 'PromoOpen']
        categorical_features = ['StateHoliday', 'StoreType', 'Assortment', 'MonthPart']

        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))])

        return pipeline

    @staticmethod
    def train_model(df):
        X = df.drop('Sales', axis=1)
        y = df['Sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = ModelBuilder.build_pipeline()
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")

        return pipeline

    @staticmethod
    def get_feature_importance(model, feature_names):
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            importances = model.named_steps['regressor'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            print("Feature Importances:")
            print(feature_importance_df.head(10))

            # Plot feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.gca().invert_yaxis()
            plt.show()
        else:
            print("The model does not support feature importances.")

# Serialization functions
class ModelSerializer:
    @staticmethod
    def serialize_model(model, save_path):
        os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = os.path.join(save_path, f"model_{timestamp}.pkl")
        joblib.dump(model, filename)
        print(f"Model saved as {filename}")
