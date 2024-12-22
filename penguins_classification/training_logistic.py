import pickle
from sklearn.linear_model import LogisticRegression
from penguins_classification.data_preprocessing import PenguinsDataProcessor

def train_logistic():
    processor = PenguinsDataProcessor()
    df = processor.load_and_clean()
    df_train, df_test, y_train, y_test = processor.split_train_test(df)

    X_train = processor.fit_transform(df_train)

    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(X_train, y_train)

    X_test = processor.transform(df_test)
    acc = lr_model.score(X_test, y_test)
    print(f"Accuracy logistic: {acc:.3f}")

    with open("models/lr_model.pck", "wb") as f:
        pickle.dump((processor.dv, processor.scaler, lr_model), f)
    print("Model logistic guardat a models/lr_model.pck")

if __name__ == "__main__":
    train_logistic()