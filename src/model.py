from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.feature_extraction import extract_features_from_data

def train_model(train_data, train_labels, test_data, test_labels):
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    X_train, X_test = extract_features_from_data(train_data, test_data)

    rf_classifier.fit(X_train, train_labels)
    y_pred = rf_classifier.predict(X_test)

    # Print results
    print(f"Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred))