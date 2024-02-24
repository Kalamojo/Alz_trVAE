import subprocess
import sys

TRAINING_DICT = {
    "trVAE": ["alzPro", "alzPro-time"],
}


def train(method, dataset, combination):
    if TRAINING_DICT.keys().__contains__(method):
        if TRAINING_DICT[method].__contains__(dataset):
            if combination in ["I", "U"]:
                command = f"python -m scripts.train_{method} {dataset} {combination}"
                subprocess.call([command], shell=True)


def main():
    if len(sys.argv) < 2:
        model_to_train = "all"
        dataset_to_train = None
    elif len(sys) < 3:
        raise Exception("Combination parameter missing")
    else:
        model_to_train = sys.argv[1]
        dataset_to_train = sys.argv[2]
        combination_method = sys.argv[3]
    if model_to_train == "all":
        train('trVAE', 'alzPro', 'I')
        train('trVAE', 'alzPro', 'U')
        train('trVAE', 'alzPro-time', 'I')
        train('trVAE', 'alzPro-time', 'U')
    else:
        train(model_to_train, dataset_to_train, combination_method)


if __name__ == '__main__':
    main()
