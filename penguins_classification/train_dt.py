import pickle
from sklearn.tree import DecisionTreeClassifier
from penguins_classification.data_preprocessing import PenguinsDataProcessor

def train_dt():
    processor = PenguinsDataProcessor()
    df = processor.load_and_clean()
    df_train, df_test, y_train, y_test = processor.split_train_test(df)

    X_train = processor.fit_transform(df_train)

    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    X_test = processor.transform(df_test)
    acc = dt_model.score(X_test, y_test)
    print(f"Accuracy (DecisionTree): {acc:.3f}")

    with open("models/dt_model.pck", "wb") as f:
        pickle.dump((processor.dv, processor.scaler, dt_model), f)
    print("Model Decision Tree guardat a models/dt_model.pck")

if __name__ == "__main__":
    train_dt()