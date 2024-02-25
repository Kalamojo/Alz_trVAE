import subprocess
import sys

TRAINING_DICT = {
    "trVAE": ["alzPro", "alzPro-time"],
}


def train(method, dataset, combination, cellType = None):
    if TRAINING_DICT.keys().__contains__(method):
        if TRAINING_DICT[method].__contains__(dataset):
            if combination in ["I", "U"]:
                if cellType:
                    command = f"python -m scripts.train_{method} {dataset} {combination} {cellType}"
                else:
                    command = f"python -m scripts.train_{method} {dataset} {combination}"
                subprocess.call([command], shell=True)


def main():
    if len(sys.argv) < 2:
        model_to_train = "all"
        dataset_to_train = None
        combination_method = None
        cell_type = None
    elif len(sys.argv) < 3:
        raise Exception("Dataset parameter missing")
    elif len(sys.argv) < 4:
        raise Exception("Combination parameter missing")
    else:
        model_to_train = sys.argv[1]
        dataset_to_train = sys.argv[2]
        combination_method = sys.argv[3]
        cell_type = sys.argv[4] if len(sys.argv) == 5 else None
    if model_to_train == "all":
        train('trVAE', 'alzPro', 'I')
        train('trVAE', 'alzPro', 'U')
        train('trVAE', 'alzPro-time', 'I')
        train('trVAE', 'alzPro-time', 'U')
    else:
        train(model_to_train, dataset_to_train, combination_method, cell_type)


if __name__ == '__main__':
    main()
