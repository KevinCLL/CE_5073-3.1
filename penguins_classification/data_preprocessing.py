import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

class PenguinsDataProcessor:
    def __init__(self):
        self.dv = None
        self.scaler = None
        self.num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
        self.cat_cols = ["island", "sex"]
    
    def load_and_clean(self):
        df = sns.load_dataset("penguins")
        df.dropna(inplace=True)
        # Convertim species a numèric
        species_map = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
        df["species"] = df["species"].map(species_map)
        return df

    def split_train_test(self, df, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
        y_train = df_train["species"].values
        y_test = df_test["species"].values
        df_train = df_train.drop("species", axis=1)
        df_test = df_test.drop("species", axis=1)
        return df_train, df_test, y_train, y_test

    def fit_transform(self, df):
        """
        Converteix df a matriu (one-hot + scaled).
        Guarda dv i scaler interns per poder reutilitzar transform().
        """
        train_dict = df[self.cat_cols + self.num_cols].to_dict(orient='records')
        self.dv = DictVectorizer(sparse=False)
        X_cat_num = self.dv.fit_transform(train_dict)

        feature_names = self.dv.get_feature_names_out()
        num_indexes = []
        cat_indexes = []
        for i, feat in enumerate(feature_names):
            if feat in self.num_cols:
                num_indexes.append(i)
            else:
                cat_indexes.append(i)

        X_num_only = X_cat_num[:, num_indexes]
        self.scaler = StandardScaler()
        X_num_scaled = self.scaler.fit_transform(X_num_only)

        import numpy as np
        X_final = np.zeros_like(X_cat_num)
        X_final[:, num_indexes] = X_num_scaled
        X_final[:, cat_indexes] = X_cat_num[:, cat_indexes]
        return X_final

    def transform(self, df):
        test_dict = df[self.cat_cols + self.num_cols].to_dict(orient='records')
        X_cat_num = self.dv.transform(test_dict)

        import numpy as np
        feature_names = self.dv.get_feature_names_out()
        num_indexes = []
        cat_indexes = []
        for i, feat in enumerate(feature_names):
            if feat in self.num_cols:
                num_indexes.append(i)
            else:
                cat_indexes.append(i)

        X_num_only = X_cat_num[:, num_indexes]
        X_num_scaled = self.scaler.transform(X_num_only)
        X_final = np.zeros_like(X_cat_num)
        X_final[:, num_indexes] = X_num_scaled
        X_final[:, cat_indexes] = X_cat_num[:, cat_indexes]

        return X_final