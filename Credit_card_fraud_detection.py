from src.data import load_data, split_data
from src.models import train_model, save_model
from src.visualization import plot_confusion_matrix, plot_roc_curve

# Load the data
data_file = 'data/creditcard.csv'
df = load_data(data_file)

# Split the data
X_train, X_test, y_train, y_test = split_data(df)

# Train the model
model = train_model(X_train, y_train)

# Save the model
model_file = 'models/random_forest.pkl'
save_model(model, model_file)

# Load the model
with open(model_file, 'rb') as f:
    loaded_model = pickle.load(f)

# Make predictions
y_pred = loaded_model.predict(X_test)
y_pred_proba = loaded_model.predict_proba(X_test)

# Visualize results
plot_confusion_matrix(y_test, y_pred)
plot_roc_curve(y_test, y_pred_proba)

