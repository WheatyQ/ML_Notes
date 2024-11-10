from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train a classifier (Logistic Regression in this case)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Step 4: Get predicted probabilities for the positive class
y_scores = clf.predict_proba(X_test)[:, 1]  # Scores/probabilities for class 1

# Step 5: Compute the Precision-Recall curve and the PR AUC
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

# Step 6: Print PR AUC result
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Step 7: Plot the Precision-Recall curve
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
