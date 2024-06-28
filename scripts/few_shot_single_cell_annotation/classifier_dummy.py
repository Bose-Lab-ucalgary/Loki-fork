def parse_args():
    """
    Parse command line arguments
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test_case", type=str, required=True)
    parser.add_argument("--modality", type=str, default="text")
    parser.add_argument("--label_col", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=2024)
    parser.add_argument("--plot_umap", action="store_true")
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    return args


# use a linear classifier to classify the data
def train_dummy(labels, verbose=True, **kwargs):
    import numpy as np
    from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
    from sklearn.dummy import DummyClassifier

    clf = DummyClassifier(**kwargs)
    dummy_X_train = np.zeros((len(labels), 1))
    clf.fit(dummy_X_train, labels)

    # using a linear classifier
    preds = clf.predict(dummy_X_train)

    accuracy = balanced_accuracy_score(labels, preds)
    accuracy_unweighted = accuracy_score(labels, preds)

    f1 = f1_score(labels, preds, average="weighted")
    f1_unweighted = f1_score(labels, preds, average="macro")

    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")

    return clf, labels, preds, f1, accuracy, f1_unweighted, accuracy_unweighted


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap="Blues", save_path=None
):
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    if not title:
        title = "Normalized confusion matrix" if normalize else "Confusion matrix"

    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()

    plt.savefig(save_path)

    plt.show()

def sample_n_data_per_class(labels, n, to_exclude_list=[]):
    """
    Randomly sample n data points from each class

    Args:
    labels: np.array, labels of the data
    n: int, number of data points to sample from each class
    to_exclude_list: list, indices of data points to exclude from sampling

    Returns:
    the indices of the sampled data
    """
    import numpy as np

    sampled_indices = []
    for label in set(labels):
        indices = np.where(labels == label)[0]
        indices = np.setdiff1d(indices, to_exclude_list)
        try:
            sampled_indices_class = np.random.choice(indices, n, replace=False)
        except ValueError:
            sampled_indices_class = np.random.choice(indices, n, replace=True)
            print(
                f"Class {label} has less than {n} data points. Sampling with replacement."
            )

        sampled_indices.extend(sampled_indices_class)

    return sampled_indices


# plot umap as scatter plot, color by annotations (a list of strings)
def scatter_plot(embedding, annotations, save_path=None, **kwargs):

    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure()
    for i, label in enumerate(set(annotations)):
        indices = np.where(np.array(annotations) == label)[0]
        plt.scatter(embedding[indices, 0], embedding[indices, 1], label=label, **kwargs)

    # place the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), markerscale=4., loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def get_umap(embedding, random_state=2024):
    import umap
    from sklearn.preprocessing import StandardScaler

    reducer = umap.UMAP(random_state=random_state)
    scaled_data = StandardScaler().fit_transform(embedding)
    embedding = reducer.fit_transform(scaled_data)
    return embedding


def main():
    args = parse_args()

    import os
    import pickle
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score, balanced_accuracy_score
    from sklearn.linear_model import SGDClassifier


    MODEL = args.model
    TEST_CASE = args.test_case
    MODALITY = args.modality
    LABEL_COL = args.label_col
    RANDOM_STATE = args.random_state
    plot_umap = args.plot_umap
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    NAME = f"{MODEL}-{TEST_CASE}-{MODALITY}"

    annotations = pd.read_csv(os.path.join(f"{TEST_CASE}", f"{TEST_CASE}_labels.csv"), sep="\t")[LABEL_COL].tolist()

    # Convert annotations to integer labels
    if os.path.exists(f"{TEST_CASE}-label_encoder.pkl"):
        with open(f"{TEST_CASE}-label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        labels = encoder.transform(annotations)
    else:
        encoder = LabelEncoder()
        labels = encoder.fit_transform(annotations)

        # Save the label encoder
        with open(f"{TEST_CASE}-label_encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)

    # Initialize dictionaries for F1 and accuracy scores
    test_f_scores = {}
    test_accuracy_scores = {}
    test_f_scores_unweighted = {}
    test_accuracy_scores_unweighted = {}

    # Train models with different number of labeled data
    # The strategy is to randomly sample a specified number of data in each class as the training data
    all_preds = []
    all_truth = []

    total_data = len(labels)
    total_classes = len(set(labels))
    label_per_class_list = [0]

    for label_per_class in tqdm(label_per_class_list, desc="training labels per class"):
        indices_used = []

        preds_per_config = []
        truth_per_config = []
        for random_seed in tqdm(range(1, 11), desc="run"):

            _, test_truth, preds, f1, accuracy, f1_unweighted, accuracy_unweighted = train_dummy(
                labels,
                verbose=False,
                random_state=random_seed,
                #strategy="stratified",
                strategy="uniform",
            )

            preds_per_config.extend(preds)
            truth_per_config.extend(test_truth)
            all_preds.extend(preds)
            all_truth.extend(test_truth)

            test_f_scores.setdefault(label_per_class, []).append(f1)
            test_accuracy_scores.setdefault(label_per_class, []).append(accuracy)
            test_f_scores_unweighted.setdefault(label_per_class, []).append(f1_unweighted)
            test_accuracy_scores_unweighted.setdefault(label_per_class, []).append(accuracy_unweighted)

    # Plot confusion matrix
    plot_confusion_matrix(
        all_truth,
        all_preds,
        encoder.classes_,
        normalize=True,
        title=f"Confusion Matrix",
        save_path=os.path.join(output_dir, f"{NAME}-confusion_matrix.png"),
    )
    # Save scores to CSV files
    pd.DataFrame(test_accuracy_scores).to_csv(
        os.path.join(output_dir, f"{NAME}-accuracy_scores.csv"), index=False
    )
    pd.DataFrame(test_f_scores).to_csv(
        os.path.join(output_dir, f"{NAME}-f1_scores.csv"), index=False
    )
    pd.DataFrame(test_accuracy_scores_unweighted).to_csv(
        os.path.join(output_dir, f"{NAME}-accuracy_scores_unweighted.csv"), index=False
    )
    pd.DataFrame(test_f_scores_unweighted).to_csv(
        os.path.join(output_dir, f"{NAME}-f1_scores_unweighted.csv"), index=False
    )

    # Plot accuracy scores
    sns.set_theme(style="ticks")

    def plot_scores(scores, ylabel, title, save_path):
        df = pd.DataFrame(scores)
        plt.figure(figsize=(5, 5))
        sns.barplot(data=df)
        plt.xlabel("Train Ratio")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(0.0, 1)
        plt.yticks(np.arange(0.0, 1, 0.2))
        plt.tight_layout()
        plt.savefig(save_path)

    plot_scores(
        test_accuracy_scores,
        "Accuracy Score",
        "Accuracy Score vs Train Ratio",
        os.path.join(output_dir, f"{NAME}-accuracy.png"),
    )
    plot_scores(
        test_f_scores,
        "F1 Score",
        "F1 Score vs Train Ratio",
        os.path.join(output_dir, f"{NAME}-f1.png"),
    )


if __name__ == "__main__":
    main()
