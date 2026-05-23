from uniad import build_uniad, load_checkpoint


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    uniad = build_uniad()
    result = load_checkpoint(uniad, args.checkpoint, strict=False)
    print(f"missing_keys={len(result['missing_keys'])}")
    print(f"unexpected_keys={len(result['unexpected_keys'])}")
    if result["meta"]:
        print(f"meta_keys={sorted(result['meta'].keys())}")


if __name__ == "__main__":
    main()
