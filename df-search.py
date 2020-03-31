# df-search.py
# Built-in libraries
from timeit import timeit

# Third-party libraries
import numpy as np
import pandas as pd


def test_query(orig, data):
    print('\n' + "-"*80)
    for index, record in data.iterrows():
        addr = record["addr"]
        result = orig.query(f'addr == {addr}')
        if not result.empty:
            print("My start addr is:", addr)
            print(result)
    print("-"*80 + '\n')


def test_mask(orig, data):
    print('\n' + "-"*80)
    for index, record in data.iterrows():
        addr = record["addr"]
        result = orig.loc[orig["addr"] == addr]
        if not result.empty:
            print("My start addr is:", addr)
            print(result)
    print("-"*80 + '\n')


def get_addr_mask(orig, addr):
    result = orig.loc[orig["addr"] == addr]
    if result.empty:
        return 0
    if result["addr"].size > 1:
        raise Exception(f"ERROR:  More than one match for artifact at addr {addr}")
    return result["addr"].iloc[0]


def test_mask_vectorized(orig, data):
    print('\n' + "-"*80)
    data["orig_addr"] = data["addr"].apply(lambda addr: get_addr_mask(orig, addr))
    print(data.iloc[:4, :])
    print("-"*80 + '\n')


def main():
    orig = pd.DataFrame({
        "type": ["xpub", "json", "xpub", "json", "ipc"] + ["foo"] * 1000,
        "addr": [100, 104, 92, 84, 96] + list(np.random.randint(0, 2, 1000))
    })

    data = pd.DataFrame({
        "type": ["json", "xpub", "json", "ipc"] + ["oof"] * 1000,
        "addr": [104, 92, 84, 96] + list(np.random.randint(2, 4, 1000))
    })

    test_mask_vectorized(orig, data)

    # Run test with DataFrame.query
    time_query = timeit(lambda: test_query(orig, data), number=5)

    # Run test with masking
    time_mask = timeit(lambda: test_mask(orig, data), number=5)

    # Run test with masking
    time_mask_vectorized = timeit(lambda: test_mask_vectorized(orig, data), number=5)

    print(f"{time_query:0.2f}\t(seconds using query)")
    print(f"{time_mask:0.2f}\t(seconds using masking)")
    print(f"{time_mask_vectorized:0.2f}\t(seconds using vectorized masking)")


if __name__ == "__main__":
    main()
