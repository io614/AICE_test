import argparse
import mlp

extract = mlp.setup.extract
preprocess = mlp.preprocessing.preprocess

# parse arguments
parser = argparse.ArgumentParser()

# flags
parser.add_argument("-t", "--test", help="run model on test set and generate evaluation metrics", action="store_true")
parser.add_argument("-l", "--log", help="log results in log.json", action="store_true")

# preview modes (mutually exclusive)
group = parser.add_mutually_exclusive_group()
group.add_argument("-p", "--preview", help="Preview raw data extracted from database (part 1 of assessment). Display first N rows (default 5)", nargs="?",
                    type=int, const=5, metavar="N")

group.add_argument("-c", "--processed", help="Preview data that has been cleaned and augmented with extra features. Display first N rows (default 5)", nargs="?",
                    type=int, const=5, metavar="N")

args = parser.parse_args()

# print(args)

# run preview modes (if selected)
if args.preview:
    print(extract().head(args.preview))

elif args.processed:
    print(extract().pipe(preprocess).head(args.processed))