import argparse
import json
import os
import mlp
from mlp.Pipeline import Pipeline

# log file
log_file_name = "log.json"


def log_to_json(dict_to_append):
    """
    function to log model params and evaluate metrics to a json file. If file does not exist, it is created
    """
    # if os.path.exists(log_file_name):
    try: 
        with open(log_file_name, "r") as read_file:
            data = json.load(read_file)
    except:
        data = []


    data.append(dict_to_append)

    with open(log_file_name, "w") as write_file:
        json.dump(data, write_file, indent=2)


# parse arguments
parser = argparse.ArgumentParser()

# flags
parser.add_argument("-t", "--test", help="Evaluate test set and generate evaluation metrics", action="store_true")
parser.add_argument("-l", "--log", help="Log params and results in log.json. If such a file does not exist, it is created.", action="store_true")

# modes (mutually exclusive)

group = parser.add_mutually_exclusive_group()

group.add_argument("-r", "--peek_raw", help="Preview raw data extracted from database (part 1 of assessment). Display first N rows (default 5)", nargs="?",
                    type=int, const=5, metavar="N")

group.add_argument("-p", "--peek_training", help="Preview training data that has been cleaned and augmented with extra features. Display first N rows (default 5)", nargs="?",
                    type=int, const=5, metavar="N")

args = parser.parse_args()


# run preview modes (if selected)
if args.peek_raw:
    print("Previewing raw data...")
    df = mlp.setup.extract()
    print("shape:", df.shape)
    print(df.head(args.peek_raw))

elif args.peek_training:
    print("Previewing training data...")
    df = mlp.preprocessing.extract_preprocess_split()[0]
    print("shape:", df.shape)
    print(df.head(args.peek_training))

else:
    pl = Pipeline()
    pl.grid_search()

    if args.test:
        pl.eval_test()

    print("="*30)
    print("Evaluation metrics:")
    [print(f"{key} : {value}") for key, value in pl.get_combined_dict().items() if 'rmse' in key]
    
    if args.log:
        log_to_json(pl.get_combined_dict())