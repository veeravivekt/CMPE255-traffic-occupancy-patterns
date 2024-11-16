from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model(X_train, train_labels, X_test, test_labels):
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    rf_classifier.fit(X_train_reshaped, train_labels)
    y_pred = rf_classifier.predict(X_test_reshaped)

    # Print results
    print(f"Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred))