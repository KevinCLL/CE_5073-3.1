from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask("penguins-classification")

with open("models/lr_model.pck", "rb") as f:
    dv_lr, scaler_lr, lr_model = pickle.load(f)

with open("models/svm_model.pck", "rb") as f:
    dv_svm, scaler_svm, svm_model = pickle.load(f)

with open("models/dt_model.pck", "rb") as f:
    dv_dt, scaler_dt, dt_model = pickle.load(f)

with open("models/knn_model.pck", "rb") as f:
    dv_knn, scaler_knn, knn_model = pickle.load(f)

def transform_single(penguin, dv, scaler):
    x_cat_num = dv.transform([penguin])
    feature_names = dv.get_feature_names_out()

    num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    num_indexes = []
    cat_indexes = []
    for i, feat in enumerate(feature_names):
        if feat in num_cols:
            num_indexes.append(i)
        else:
            cat_indexes.append(i)

    X_num_only = x_cat_num[:, num_indexes]
    X_num_scaled = scaler.transform(X_num_only)

    X_final = np.zeros_like(x_cat_num)
    X_final[:, num_indexes] = X_num_scaled
    X_final[:, cat_indexes] = x_cat_num[:, cat_indexes]

    return X_final

@app.route("/predict_lr", methods=["POST"])
def predict_lr():
    penguin = request.get_json()
    X = transform_single(penguin, dv_lr, scaler_lr)
    pred = lr_model.predict(X)[0]  # 0,1,2
    return jsonify({"model": "LogisticRegression", "prediction": int(pred)})

@app.route("/predict_svm", methods=["POST"])
def predict_svm():
    penguin = request.get_json()
    X = transform_single(penguin, dv_svm, scaler_svm)
    pred = svm_model.predict(X)[0]
    return jsonify({"model": "SVM", "prediction": int(pred)})

@app.route("/predict_dt", methods=["POST"])
def predict_dt():
    penguin = request.get_json()
    X = transform_single(penguin, dv_dt, scaler_dt)
    pred = dt_model.predict(X)[0]
    return jsonify({"model": "DecisionTree", "prediction": int(pred)})

@app.route("/predict_knn", methods=["POST"])
def predict_knn():
    penguin = request.get_json()
    X = transform_single(penguin, dv_knn, scaler_knn)
    pred = knn_model.predict(X)[0]
    return jsonify({"model": "KNN", "prediction": int(pred)})

if __name__ == "__main__":
    app.run(debug=True, port=8000)