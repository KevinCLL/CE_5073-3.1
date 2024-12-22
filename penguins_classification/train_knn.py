import pickle
from sklearn.neighbors import KNeighborsClassifier
from penguins_classification.data_preprocessing import PenguinsDataProcessor

def train_knn():
    processor = PenguinsDataProcessor()
    df = processor.load_and_clean()
    df_train, df_test, y_train, y_test = processor.split_train_test(df)

    X_train = processor.fit_transform(df_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    X_test = processor.transform(df_test)
    acc = knn_model.score(X_test, y_test)
    print(f"Accuracy (KNN): {acc:.3f}")

    with open("models/knn_model.pck", "wb") as f:
        pickle.dump((processor.dv, processor.scaler, knn_model), f)
    print("Model KNN guardat a models/knn_model.pck")

if __name__ == "__main__":
    train_knn()