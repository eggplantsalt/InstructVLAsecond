import json
from pathlib import Path

paths = [
    Path("data_pipeline/data/bridge_instruction.json"),
    Path("data_pipeline/data/fractal_instruction.json"),
]

for path in paths:
    print("\n" + "=" * 100)
    print("FILE:", path)
    print("EXISTS:", path.exists())
    if not path.exists():
        continue

    print("SIZE_MB:", round(path.stat().st_size / 1024 / 1024, 2))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("TOP_TYPE:", type(data).__name__)

    if isinstance(data, dict):
        print("NUM_TOP_KEYS:", len(data))
        keys = list(data.keys())
        print("FIRST_20_KEYS:", keys[:20])

        first_key = keys[0] if keys else None
        print("FIRST_KEY:", first_key)

        sample = data[first_key] if first_key is not None else None

    elif isinstance(data, list):
        print("NUM_ITEMS:", len(data))
        sample = data[0] if data else None

    else:
        sample = data

    print("\nSAMPLE_TYPE:", type(sample).__name__)

    if isinstance(sample, dict):
        print("SAMPLE_KEYS:", list(sample.keys()))
    elif isinstance(sample, list):
        print("SAMPLE_LIST_LEN:", len(sample))
        if sample:
            print("SAMPLE_0_TYPE:", type(sample[0]).__name__)
            if isinstance(sample[0], dict):
                print("SAMPLE_0_KEYS:", list(sample[0].keys()))

    print("\nSAMPLE_PREVIEW:")
    try:
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:3000])
    except Exception as e:
        print("Could not dump sample:", repr(e))
