from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and predict
model.load_weights("best_model.h5")
predictions = model.predict(val_generator)
y_pred = predictions.argmax(axis=1)
y_true = val_generator.classes

print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.show()
