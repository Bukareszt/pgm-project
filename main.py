import naive_bayes
import load_data
from sklearn import metrics as sk_mtr

def main():
    dataset = load_data.load_iris_dataset()
    clf = naive_bayes.GaussianNBClassifier(num_epochs=20)
    clf.fit(X=dataset["train"]["X"], y=dataset["train"]["y"])

    for split in ("train", "test"):
        print(split)
        print(sk_mtr.classification_report(
            y_true=dataset[split]["y"],
            y_pred=clf.predict(X=dataset[split]["X"]),
        ))

main()