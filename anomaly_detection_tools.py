from smolagents import Tool
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json


class LoadDatasetTool(Tool):
    name = "load_dataset"
    description = """
    Loads a dataset for anomaly detection. Can load sample datasets or a CSV/Excel file from a specified path.
    Returns a pandas DataFrame with the loaded data.
    """
    inputs = {
        "dataset_name": {
            "type": "string",
            "description": "Name of the sample dataset ('Credit Card Transactions', 'IoT Sensor Data', 'Network Logs') or None if using file_path",
            "nullable": True
        },
        "file_path": {
            "type": "string",
            "description": "Absolute path to the uploaded CSV or Excel file. Required if dataset_name is None",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, dataset_name: str = None, file_path: str = None):
        try:
            if file_path:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

                # Check if DataFrame is empty
                if df.empty:
                    raise ValueError("Loaded DataFrame is empty")

                return df.to_json(orient='records', date_format='iso')

            elif dataset_name:
                if dataset_name == "Credit Card Transactions":
                    dataset = pd.read_csv("credit_card_fraud_dataset.csv")
                    df = dataset.rename(columns={'IsFraud': 'label'})

                elif dataset_name == "IoT Sensor Data":
                    dataset = pd.read_csv("Occupancy.csv")
                    df = dataset.rename(columns={'Occupancy': 'label'})

                elif dataset_name == "Network Logs":
                    dataset = pd.read_csv("cybersecurity_intrusion_data.csv")
                    df = dataset.rename(columns={'attack_detected': 'label'})
                else:
                    raise ValueError(f"Unknown sample dataset: {dataset_name}")

                return df.to_json(orient='records', date_format='iso')
            else:
                raise ValueError("Either dataset_name or file_path must be provided.")

        except Exception as e:
            return json.dumps({"error": f"Failed to load dataset: {str(e)}"})


class PreprocessDataTool(Tool):
    name = "preprocess_data"
    description = """
    Preprocesses a dataset by handling missing values, encoding categorical variables, 
    separating features and labels, and scaling numerical features.
    Returns a dictionary with processed data.
    """
    inputs = {
        "df_json": {
            "type": "string",
            "description": "JSON string representation of the pandas DataFrame to preprocess"
        }
    }
    output_type = "string"

    def forward(self, df_json: str):
        try:
            # Convert JSON back to DataFrame
            df = pd.read_json(df_json, orient='records')
            df_processed = df.copy()

            # Handle missing values
            for col in df_processed.columns:
                if df_processed[col].dtype in ['object', 'category']:
                    if not df_processed[col].mode().empty:
                        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                    else:
                        df_processed[col].fillna('Unknown', inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)

            # Handle categorical variables
            categorical_cols = df_processed.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                le = LabelEncoder()
                for col in categorical_cols:
                    if col != 'label':  # Don't encode the label column
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))

            # Separate features and labels
            if 'label' in df_processed.columns:
                X = df_processed.drop('label', axis=1)
                y = df_processed['label'].values
            else:
                X = df_processed
                y = None

            # Scale numerical features
            numerical_cols = X.select_dtypes(include=[np.number]).columns

            # Create a copy for scaling
            X_scaled = X.copy()

            if len(numerical_cols) > 0:
                scaler = StandardScaler()
                # Scale only numerical columns
                X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])

            # Return as JSON
            result = {
                "X": X.to_json(orient='records'),
                "X_scaled": X_scaled.to_json(orient='records'),
                "y": y.tolist() if y is not None else None,
                "feature_names": X.columns.tolist(),
                "scaled_feature_names": X_scaled.columns.tolist(),
                "status": "success"
            }

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"Preprocessing failed: {str(e)}", "status": "error"})


class IsolationForestTool(Tool):
    name = "isolation_forest_detector"
    description = """
    Applies the Isolation Forest algorithm for anomaly detection.
    Returns predictions and anomaly scores.
    """
    inputs = {
        "X_scaled": {
            "type": "string",
            "description": "JSON string of scaled feature data"
        },
        "y": {
            "type": "string",
            "description": "JSON string of true labels (list) or null if unsupervised",
            "nullable": True
        },
        "n_estimators": {
            "type": "integer",
            "description": "The number of base estimators in the ensemble"
        },
        "contamination": {
            "type": "number",
            "description": "The proportion of outliers in the data set"
        }
    }
    output_type = "string"

    def forward(self, X_scaled: str, n_estimators: int, contamination: float, y: str = None):
        try:
            # Parse inputs - convert JSON to DataFrame first, then to numpy array
            X_scaled_df = pd.read_json(X_scaled, orient='records')
            X_scaled_array = X_scaled_df.values

            y_array = None
            if y and y != "null" and y != "None":
                y_array = np.array(json.loads(y))

            # Apply Isolation Forest
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42
            )

            predictions = model.fit_predict(X_scaled_array)
            scores = model.score_samples(X_scaled_array)
            predictions_binary = np.where(predictions == -1, 1, 0)

            # Calculate metrics if true labels are available
            metrics = {}
            if y_array is not None and len(y_array) > 0:
                metrics = {
                    "accuracy": float(accuracy_score(y_array, predictions_binary)),
                    "precision": float(precision_score(y_array, predictions_binary, zero_division=0)),
                    "recall": float(recall_score(y_array, predictions_binary, zero_division=0)),
                    "f1_score": float(f1_score(y_array, predictions_binary, zero_division=0))
                }

            result = {
                "predictions": predictions_binary.tolist(),
                "scores": scores.tolist(),
                "metrics": metrics,
                "anomaly_count": int(np.sum(predictions_binary)),
                "total_samples": len(predictions_binary),
                "status": "success"
            }

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"Isolation Forest failed: {str(e)}", "status": "error"})


class LocalOutlierFactorTool(Tool):
    name = "local_outlier_factor_detector"
    description = """
    Applies the Local Outlier Factor algorithm for anomaly detection.
    Returns predictions and outlier scores.
    """
    inputs = {
        "X_scaled": {
            "type": "string",
            "description": "JSON string of scaled feature data"
        },
        "y": {
            "type": "string",
            "description": "JSON string of true labels (list) or null if unsupervised",
            "nullable": True
        },
        "n_neighbors": {
            "type": "integer",
            "description": "Number of neighbors to use for kneighbors queries"
        },
        "contamination": {
            "type": "number",
            "description": "The proportion of outliers in the data set"
        }
    }
    output_type = "string"

    def forward(self, X_scaled: str, n_neighbors: int, contamination: float, y: str = None):
        try:
            # Parse inputs
            X_scaled_df = pd.read_json(X_scaled, orient='records')
            X_scaled_array = X_scaled_df.values

            y_array = None
            if y and y != "null" and y != "None":
                y_array = np.array(json.loads(y))

            # Apply Local Outlier Factor
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=False
            )

            predictions = model.fit_predict(X_scaled_array)
            scores = model.negative_outlier_factor_
            predictions_binary = np.where(predictions == -1, 1, 0)

            # Calculate metrics if true labels are available
            metrics = {}
            if y_array is not None and len(y_array) > 0:
                metrics = {
                    "accuracy": float(accuracy_score(y_array, predictions_binary)),
                    "precision": float(precision_score(y_array, predictions_binary, zero_division=0)),
                    "recall": float(recall_score(y_array, predictions_binary, zero_division=0)),
                    "f1_score": float(f1_score(y_array, predictions_binary, zero_division=0))
                }

            result = {
                "predictions": predictions_binary.tolist(),
                "scores": scores.tolist(),
                "metrics": metrics,
                "anomaly_count": int(np.sum(predictions_binary)),
                "total_samples": len(predictions_binary),
                "status": "success"
            }

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"Local Outlier Factor failed: {str(e)}", "status": "error"})


class OneClassSVMTool(Tool):
    name = "one_class_svm_detector"
    description = """
    Applies the One-Class SVM algorithm for anomaly detection.
    Returns predictions and decision function scores.
    """
    inputs = {
        "X_scaled": {
            "type": "string",
            "description": "JSON string of scaled feature data"
        },
        "y": {
            "type": "string",
            "description": "JSON string of true labels (list) or null if unsupervised",
            "nullable": True
        },
        "nu": {
            "type": "number",
            "description": "An upper bound on the fraction of training errors"
        },
        "kernel": {
            "type": "string",
            "description": "Specifies the kernel type to be used in the algorithm"
        },
        "gamma": {
            "type": "string",
            "description": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'"
        }
    }
    output_type = "string"

    def forward(self, X_scaled: str, nu: float, kernel: str, gamma: str, y: str = None):
        try:
            # Parse inputs
            X_scaled_df = pd.read_json(X_scaled, orient='records')
            X_scaled_array = X_scaled_df.values

            y_array = None
            if y and y != "null" and y != "None":
                y_array = np.array(json.loads(y))

            # Apply One-Class SVM
            model = OneClassSVM(
                nu=nu,
                kernel=kernel,
                gamma=gamma
            )

            predictions = model.fit_predict(X_scaled_array)
            scores = model.score_samples(X_scaled_array)
            predictions_binary = np.where(predictions == -1, 1, 0)

            # Calculate metrics if true labels are available
            metrics = {}
            if y_array is not None and len(y_array) > 0:
                metrics = {
                    "accuracy": float(accuracy_score(y_array, predictions_binary)),
                    "precision": float(precision_score(y_array, predictions_binary, zero_division=0)),
                    "recall": float(recall_score(y_array, predictions_binary, zero_division=0)),
                    "f1_score": float(f1_score(y_array, predictions_binary, zero_division=0))
                }

            result = {
                "predictions": predictions_binary.tolist(),
                "scores": scores.tolist(),
                "metrics": metrics,
                "anomaly_count": int(np.sum(predictions_binary)),
                "total_samples": len(predictions_binary),
                "status": "success"
            }

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"One-Class SVM failed: {str(e)}", "status": "error"})


class DataOverviewTool(Tool):
    name = "data_overview"
    description = """
    Provides a comprehensive overview of a dataset including basic statistics,
    data types, missing values, and summary information.
    """
    inputs = {
        "df_json": {
            "type": "string",
            "description": "JSON string representation of the pandas DataFrame to analyze"
        }
    }
    output_type = "string"

    def forward(self, df_json: str):
        try:
            # Convert JSON back to DataFrame
            df = pd.read_json(df_json, orient='records')

            # Basic statistics
            overview = {
                "total_records": len(df),
                "total_features": len(df.columns) - (1 if 'label' in df.columns else 0),
                "missing_values": int(df.isnull().sum().sum()),
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "unique_counts": df.nunique().to_dict(),
                "status": "success"
            }

            # Anomaly rate if labels are present
            if 'label' in df.columns:
                anomaly_rate = (df['label'].sum() / len(df)) * 100
                overview["anomaly_rate"] = float(anomaly_rate)

            # Numerical statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                overview["numerical_statistics"] = df[numeric_cols].describe().to_dict()

            # Categorical value counts
            categorical_cols = df.select_dtypes(include=['object']).columns
            categorical_info = {}
            for col in categorical_cols:
                categorical_info[col] = df[col].value_counts().head(10).to_dict()
            overview["categorical_info"] = categorical_info

            return json.dumps(overview)

        except Exception as e:
            return json.dumps({"error": f"Data overview failed: {str(e)}", "status": "error"})


class ReportGeneratorTool(Tool):
    name = "generate_anomaly_report"
    description = """
    Generates a comprehensive report from anomaly detection results including
    metrics, summary statistics, and recommendations.
    """
    inputs = {
        "algorithm_name": {
            "type": "string",
            "description": "Name of the anomaly detection algorithm used"
        },
        "results_json": {
            "type": "string",
            "description": "JSON string of the anomaly detection results"
        },
        "data_overview_json": {
            "type": "string",
            "description": "JSON string of the data overview"
        }
    }
    output_type = "string"

    def forward(self, algorithm_name: str, results_json: str, data_overview_json: str):
        try:
            results = json.loads(results_json)
            overview = json.loads(data_overview_json)

            # Check for errors
            if "error" in results:
                return f"Error in results: {results['error']}"
            if "error" in overview:
                return f"Error in overview: {overview['error']}"

            # Generate report
            report = f"""
# Anomaly Detection Report

## Algorithm: {algorithm_name}

## Dataset Overview
- Total Records: {overview.get('total_records', 'N/A')}
- Total Features: {overview.get('total_features', 'N/A')}
- Missing Values: {overview.get('missing_values', 'N/A')}
- Anomaly Rate (if labeled): {overview.get('anomaly_rate', 'N/A'):.2f}%

## Detection Results
- Total Samples Analyzed: {results.get('total_samples', 'N/A')}
- Anomalies Detected: {results.get('anomaly_count', 'N/A')}
- Detection Rate: {(results.get('anomaly_count', 0) / results.get('total_samples', 1) * 100):.2f}%

## Performance Metrics (if ground truth available)
"""

            if results.get('metrics'):
                metrics = results['metrics']
                report += f"""
- Accuracy: {metrics.get('accuracy', 'N/A'):.3f}
- Precision: {metrics.get('precision', 'N/A'):.3f}
- Recall: {metrics.get('recall', 'N/A'):.3f}
- F1-Score: {metrics.get('f1_score', 'N/A'):.3f}
"""
            else:
                report += "\nNo ground truth labels available for performance evaluation."

            report += f"""

## Recommendations
1. Review the {results.get('anomaly_count', 0)} detected anomalies for further investigation
2. Consider adjusting algorithm parameters if detection rate seems too high or low
3. Validate results with domain experts
4. Consider ensemble methods for improved detection accuracy

## Technical Details
- Algorithm: {algorithm_name}
- Feature scaling: StandardScaler applied
- Missing values: Handled via median/mode imputation
- Categorical encoding: Label encoding applied
"""

            return report

        except Exception as e:
            return f"Report generation failed: {str(e)}"
