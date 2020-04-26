import argparse
import mlp
from mlp.Pipeline import Pipeline

# parse arguments
parser = argparse.ArgumentParser()

# flags
parser.add_argument("-t", "--test", help="Evaluate test set and generate evaluation metrics", action="store_true")
parser.add_argument("-l", "--log", help="log results in log.json", action="store_true")

# preview modes (mutually exclusive)
group = parser.add_mutually_exclusive_group()
group.add_argument("-p", "--preview", help="Preview raw data extracted from database (part 1 of assessment). Display first N rows (default 5)", nargs="?",
                    type=int, const=5, metavar="N")

group.add_argument("-r", "--training", help="Preview training data that has been cleaned and augmented with extra features. Display first N rows (default 5)", nargs="?",
                    type=int, const=5, metavar="N")

args = parser.parse_args()


# run preview modes (if selected)
if args.preview:
    df = mlp.setup.extract()
    print("Raw data")
    print("shape:", df.shape)
    print(df.head(args.preview))

elif args.training:
    print("training data")
    df = mlp.preprocessing.extract_preprocess_split()[0]
    print("shape:", df.shape)
    print(df.head(args.training))

else:
    pl = Pipeline()
    pl.grid_search()

    if args.test:
        pl.eval_test()

    print("="*30)
    print("Evaluation metrics:")
    [print(f"{key} : {value}") for key, value in pl.get_combined_dict().items() if 'rmse' in key]
    # if log...