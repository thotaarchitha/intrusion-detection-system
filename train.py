"""Train the intrusion detection model: python train.py"""

import argparse

from ids import pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="models/ids_model.joblib")
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()
    pipeline.run(args.data_dir, args.model_path, args.trees)
