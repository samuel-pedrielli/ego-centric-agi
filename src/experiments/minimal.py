"""
Minimal experiment runner (toy model placeholder).
"""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config")
    args = parser.parse_args()
    print(f"[minimal] placeholder run with config: {args.config}")

if __name__ == "__main__":
    main()
