from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from src.feature_extraction import extract_features_from_data

def train_model(train_data, train_labels, test_data, test_labels):
    # Define classifiers with optimized parameters
    rf_classifier = RandomForestClassifier(
        n_estimators=2000,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )

    gb_classifier = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.005,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )

    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_classifier),
            ('gb', gb_classifier)
        ],
        voting='soft',
        weights=[2, 1]
    )

    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', voting_clf)
    ])

    X_train, X_test = extract_features_from_data(train_data, test_data)

    # Train and evaluate
    pipeline.fit(X_train, train_labels)
    y_pred = pipeline.predict(X_test)

    # Print results
    print(f"Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, 
                            target_names=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']))