import pickle
from sklearn.svm import SVC
from penguins_classification.data_preprocessing import PenguinsDataProcessor

def train_svm():
    processor = PenguinsDataProcessor()
    df = processor.load_and_clean()
    df_train, df_test, y_train, y_test = processor.split_train_test(df)

    X_train = processor.fit_transform(df_train)

    svm_model = SVC()
    svm_model.fit(X_train, y_train)

    X_test = processor.transform(df_test)
    acc = svm_model.score(X_test, y_test)
    print(f"Accuracy (SVM): {acc:.3f}")

    with open("models/svm_model.pck", "wb") as f:
        pickle.dump((processor.dv, processor.scaler, svm_model), f)
    print("Model SVM guardat a models/svm_model.pck")

if __name__ == "__main__":
    train_svm()