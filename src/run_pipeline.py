\
"""
Run the full pipeline: preprocessing → scoring/AHP → target generation → train/evaluate models → save Table 11 & Figure 6.
Usage:
    python run_pipeline.py --config config.yaml
"""
import argparse
from src.utils import load_config
from src.modeling import train_and_eval_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    res = train_and_eval_all(cfg)
    print("\n=== Saved Table 11 to:", cfg['paths']['tables_dir'] + "table11_model_comparison.csv")
    print("=== Saved Figure 6 to:", cfg['paths']['figures_dir'] + "figure6_prc_by_class.png")
    print("\n", res.to_string(index=False))

if __name__ == "__main__":
    main()
