import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class FraudDetectionModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()

    def load_data(self):
        self.df = pd.read_csv(self.file_path)

        self.X = self.df[[
            "age",
            "vehicle_age",
            "prior_claims",
            "annual_miles"
        ]]

        self.y = self.df["fraud"]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42
        )

    def preprocess(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.predictions = self.model.predict(self.X_test)
        return self.predictions

    def evaluate(self):
        accuracy = accuracy_score(self.y_test, self.predictions)
        report = classification_report(self.y_test, self.predictions)

        print("Accuracy:", accuracy)
        print("\nClassification Report:\n", report)

    def run(self):
        self.load_data()
        self.split_data()
        self.preprocess()
        self.train()
        self.predict()
        self.evaluate()


if __name__ == "__main__":
    model = FraudDetectionModel("claims.csv")
    model.run()