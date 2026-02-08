import argparse
import torch
from nets import Model

def count_params(module):
    return sum(p.numel() for p in module.parameters())

def print_top_level(model):
    print("Top-level modules:")
    for name, mod in model.named_children():
        print(f"{name}: {count_params(mod):,}")
    print("-" * 40)
    print(f"Total: {count_params(model):,}")

def print_recursive(module, name=None, indent=0):
    display = name if name is not None else module.__class__.__name__
    total = count_params(module)
    print("  " * indent + f"{display}: {total:,}")
    for child_name, child in module.named_children():
        print_recursive(child, child_name, indent + 1)

def main():
    parser = argparse.ArgumentParser(description="Count parameters of CREStereo modules")
    parser.add_argument("--detailed", action="store_true", help="Print recursive detailed breakdown")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = Model()
    model.to(device)
    model.eval()

    print_top_level(model)
    if args.detailed:
        print("\nDetailed recursive breakdown:")
        print_recursive(model)

if __name__ == "__main__":
    main()
