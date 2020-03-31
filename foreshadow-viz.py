"""
foreshadow-viz.py
Visualize memory artifacts from hardware wallet clients.
By default shows artifact quantity and integrity over time.
Can optionally visualize each frame of a memory dump
"""
# Built-in libraries
import base64
import os
import os.path as osp
import sys
from datetime import datetime

# Third-party libraries
import matplotlib.cm as cm                  # Colormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Pandas config
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


def get_csvs(csv_dir, extension=".csv"):
    """
    Get names of available csv files
    :param csv_dir:     path/to/directory containing csv files
    :param extension:   csv extension
    :return:            List of strings representing names of csv files
    """
    return [filename for filename in os.listdir(csv_dir) if filename.endswith(extension)]


def transform_data(alldata, compare=False, original=None, filename=None):
    """
    Get usable data from the raw .csv data
    :param alldata:     Data loaded from .csv file
    :param compare:
    :param original:
    :return:            Relevant columns, transformed for our purposes
    """
    print(filename)
    if compare and original is None:
        raise Exception("Cannot compare DataFrame to nothing.")

    # Add a decimal start address column
    alldata["start_addr_hex"] = alldata["start_addr_hex"].apply(lambda x: x[:-1])
    alldata["start_addr_dec"] = alldata["start_addr_hex"].apply(lambda x: int(x[2:], 16))
    # Also add hex and dec columns for end address
    alldata["end_addr_dec"] = alldata["start_addr_dec"] + alldata["size_bytes"]
    alldata["end_addr_hex"] = alldata["end_addr_dec"].apply(lambda x: hex(x))
    # Decode base64-encoded data
    alldata["data"] = alldata["data_base64"].apply(lambda x: base64.standard_b64decode(x))

    def get_orig_record(orig, record, field=None):
        addr = record["start_addr_dec"]
        result = orig.loc[orig["start_addr_dec"] == addr]

        # New artifact - Return same data
        # Return empty array of original data size if it is empty
        if result.empty:
            if not field:
                return record["data"]
            if field == "integrity":
                return np.ones(len(record["data"]), dtype=np.int8)
            return record[field]

        # Previously found artifact - Return old data
        match = result.iloc[0, :]
        # Return a specific data item if a field is specified.  Otherwise, return the entire record.
        return match[field] if field else match

    if compare:
        print("\tCreating integrity vectors for new DataFrame.  ", end='')
        # Find the corresponding records in the original based on the start_addr_dec
        alldata["orig_data"] = alldata.apply(
            lambda record: get_orig_record(original, record, field="data"), axis=1
        )
        alldata["orig_integrity"] = alldata.apply(
            lambda record: get_orig_record(original, record, field="integrity"), axis=1
        )
        print("Done.")

        # (1) Bytewise and of string and previous
        #     e.g.  h e l l o   (orig_data)
        #           h e l p o   (data)
        #           1 1 1 0 1   (integrity)
        # Convert data to a np.array of bytes for fast comparison
        print("\t(1) Performing bytewise and.  ", end='')
        alldata["data_bytearray"] = alldata["data"].apply(lambda x: np.array([byt for byt in x]))
        alldata["orig_bytearray"] = alldata["orig_data"].apply(lambda x: np.array([byt for byt in x]))
        alldata["integrity"] = alldata.apply(
            lambda record: np.where(record["data_bytearray"] == record["orig_bytearray"], 1, 0), axis=1
        )
        print("Done.")

        # (2) Bitwise and of integrity and previous integrity
        #     e.g.  h e l l o       Integrity:               1 0 1 1 1
        #           h e l p o       Integrity: (step 1)      1 1 1 0 1
        #                           Integrity: (final)       1 0 1 0 1
        print("\t(2) Performing bitwise and.  ", end='')

        def do_integrity_and(record):
            new_integrity = np.where(
                np.logical_and(record["integrity"], record["orig_integrity"]),
                1, 0
            )
            return new_integrity

        alldata["integrity"] = alldata.apply(
            #lambda record: np.where(np.logical_and(record["integrity"], record["orig_integrity"]), 1, 0), axis=1
            lambda record: do_integrity_and(record), axis=1
        )
        print("Done.")

        # Print sum of integrity vector for testing - changed if sum != original length
        # alldata["sum"] = alldata["integrity"].apply(lambda x: x.sum())
        # alldata["datalen"] = alldata["data"].apply(lambda x: len(x))
        # print(alldata.groupby("sum")["data"].count())
    else:
        alldata["integrity"] = alldata["data"].apply(lambda x: np.ones(len(x), dtype=np.int8))

    # Calculate the corruption level of the memory artifact, expressed as a fraction
    # e.g. 1.0 = Not corrupted
    #      0.5 = half the bytes have changed
    #      0.0 = Every byte has changed
    print("\tCalculating corruption levels.  ", end='')

    def calc_corruption_percent(record):
        return record["integrity"].sum() / record["size_bytes"]

    alldata["corruption_level"] = alldata.apply(calc_corruption_percent, axis=1)
    print("Done.")

    # Extract relevant info =======================================================================
    return alldata[["type", "start_addr_dec", "end_addr_dec", "size_bytes", "start_addr_hex", "end_addr_hex", "data",
                    "integrity", "corruption_level"]]


def plot_data(data, frame_num):
    """
    Plot the memory artifacts in virtual memory space
    :param data:        Pandas DataFrame containing data for each relevant memory artifact
    :return:            None
    """
    # Set initial styling parameters
    sns.set(style="dark")

    # Create figure
    fig = plt.figure()

    ADDR_SCALE = 100
    ROW_SCALE = 100

    # Determine plotting parameters based on data
    # Address range
    hi_end = data["end_addr_dec"].max()
    lo_start = data["start_addr_dec"].min()
    addr_range = hi_end - lo_start
    addr_range //= ADDR_SCALE
    dim = int(np.ceil(np.sqrt(addr_range)))
    viz_rows = int(dim * 1.1)
    viz_cols = dim

    # Determine where each artifact lies on the grid
    # There are dim rows and dim cols
    # Each row contains row_size kB
    # Pad the rows in case some get cut off
    grid = np.zeros((viz_rows, viz_cols))
    row_size = addr_range // dim
    cells = row_size * dim

    for index, record in data.iterrows():
        XPUB_ROWS = 12
        XPUB_COLS = 12
        XPUB_SCALE = 4
        is_xpub = record["type"] == "xpub"

        # Determine start and stop row, same for all artifact types
        start_addr = record["start_addr_dec"] // ADDR_SCALE
        size_bytes = record["size_bytes"] // ADDR_SCALE
        row_start = start_addr // dim
        col_start = start_addr % dim
        row_end = row_start + (XPUB_ROWS * XPUB_SCALE if is_xpub else ROW_SCALE)
        col_end = col_start + (XPUB_COLS* XPUB_SCALE if is_xpub else size_bytes)
        # print(index, start_addr, size_bytes, row_start, col_start)

        # Reshape the integrity grid of the artifact from 1-D to 2-D
        # e.g. json 10200 bytes -> 100x102 array
        # e.g. xpub 144 bytes -> 24x24 array (scaled 2x)
        integrity_flat = record["integrity"]
        num_rows = XPUB_ROWS * XPUB_SCALE if is_xpub else ROW_SCALE
        num_cols = XPUB_COLS * XPUB_SCALE if is_xpub else (integrity_flat.size // ROW_SCALE)
        grid_size = num_rows * num_cols                 # Total size of integrity_grid
        # Create the new 2x2 integrity grid
        if is_xpub:
            integrity_grid = np.copy(np.reshape(integrity_flat, (XPUB_ROWS, XPUB_COLS)))
            # integrity_grid = np.random.randint(0, 2, integrity_grid.shape)
            integrity_grid = np.where(integrity_grid > 0, 0.5, 0)
            # Use the Kronecker product to scale the xpub array, e.g. 2x2 -> 4x4
            integrity_grid = np.kron(integrity_grid, np.ones((XPUB_SCALE, XPUB_SCALE)))
        else:
            shrunk_flat = integrity_flat[:grid_size]
            integrity_grid = np.copy(np.reshape(shrunk_flat, (num_rows, num_cols)))
            # integrity_grid = np.random.randint(0, 2, integrity_grid.shape)

        # Bounds check on the array
        # print("rows:", row_start, row_end, "-----  cols:", col_start, col_end)
        if row_start > viz_rows or col_start > viz_cols:
            # print("OUT OF BOUNDS - continuing")
            continue
        if row_end > viz_rows:
            new_rows = num_rows - (row_end - dim)
            integrity_grid = integrity_grid[:new_rows, :]
            # print("Truncating rows: ", integrity_grid.shape)
        if col_end > viz_cols:
            new_cols = num_cols - (col_end - dim)
            integrity_grid = integrity_grid[:, :new_cols]
            # print("Truncating cols: ", integrity_grid.shape)

        # Add artifact to the plot grid
        # grid[row_start:row_end, col_start:col_end] = 1
        try:
            grid[row_start:row_end, col_start:col_end] = integrity_grid
        except Exception as ex:
            print("======")
            print(record)
            print(integrity_grid.shape)
            print(row_end - row_start, col_end - col_start)
            print(ex)
            print(dim)
            exit()

    # grid = np.random.randint(0, 100, (dim, dim))
    # print("\nPlotting...")
    plt.imshow(grid, cmap=cm.inferno)

    # Additional styling and labeling
    plt.title("Artifacts in Physical Memory Space: " + str(frame_num))

    TICK_SCALE = 0x10000
    locs, labels = plt.xticks()
    # Labels dec -> hex e.g. 1000 -> ~0x7a120
    # [1:-1] to get rid of the default negative label and default last label
    new_labels = [hex(int(loc) * ROW_SCALE) for loc in locs]
    plt.xticks(ticks=locs[1:-1], labels=new_labels[1:-1], rotation=45)

    locs, labels = plt.yticks()
    # Labels dec -> hex e.g. 1000 -> ~0x1a4a0000
    # [1:-1] to get rid of the default negative label and default last label
    new_labels = [hex((int(loc) * dim * ROW_SCALE)) for loc in locs]
    plt.yticks(ticks=locs[1:-1], labels=new_labels[1:-1])

    # Set axis titles
    plt.xlabel("Memory Offset (From Start of Row)")
    plt.ylabel("Memory Location (Start of Row)")

    # Display a vertical side on the righthand side of the plot showing the colors corresponding to different values
    # plt.colorbar()
    plt.tight_layout(pad=3)


def plot_differential_line_graph(data, xtick_list=None):
    """
    Plot a line graph showing how the number of different artifact types in memory
    change between memory dumps.
    :param data:            List of DataFrames containing information for artifact types,
                            one DataFrame for each memory dump.
                            colums: counts_all, counts_corrupted, mean_corrupted
    :param xtick_list:      If supplied, xtick labels are set to this list.
    :return:                None
    """
    # Take list of DataFrames and convert to separate DataFrames, one for each metric
    counts_all = pd.DataFrame({
        str(i): data[i]["counts_all"] for i in range(len(data))
    }).transpose()

    counts_corr = pd.DataFrame({
        str(i): data[i]["counts_corrupted"] for i in range(len(data))
    }).transpose()

    counts_not_corr = counts_all - counts_corr

    mean_corr = pd.DataFrame({
        str(i): data[i]["mean_corrupted"] for i in range(len(data))
    }).transpose()

    print()
    print(counts_all, end='\n\n')
    print(counts_not_corr, end='\n\n')
    print(mean_corr, end='\n\n')

    # ================================================================================
    # Plot only xpub keys (much larger quantity than the rest)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    ax_xpub = axes[0]
    # sns.lineplot(data=data_xpub, sort=False, ax=ax_xpub)
    counts_all_xpub = counts_all[["xpub"]]
    # counts_all_xpub["xpub_not_corrupted"] = counts_not_corr["xpub"]
    # Plot the first line - All elements
    sns.lineplot(data=counts_all_xpub, sort=False, ax=ax_xpub)
    # Plot the second line - Uncorrupted elements
    # sns.lineplot(data=counts_not_corr_xpub, sort=False, ax=ax_xpub)
    legend = ax_xpub.legend()
    legend.set_title("Artifact Type")

    # ================================================================================
    # Plot all data other than xpub keys
    ax_rest = axes[1]
    counts_all_rest = counts_all.loc[:, counts_all.columns != "xpub"]
    sns.lineplot(data=counts_all_rest, sort=False, ax=ax_rest)
    legend = ax_rest.legend()
    legend.set_title("Artifact Type")

    # Set styling on charts
    ax_xpub.set_title("Quantity of Artifacts in Memory Across Multiple Memory Dumps", pad=40)
    for ax in [ax_xpub, ax_rest]:
        ax.set_ylabel("Number of Artifacts")
    ax_rest.set_xlabel("Time Elapsed Since First Memory Dump (H:MM:SS)")
    fig.tight_layout(pad=3)

    # ================================================================================
    # Create a new figure showing the average corruption of memory artifacts over time
    fig2, ax_corr = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
    sns.lineplot(data=mean_corr, sort=False, ax=ax_corr)

    # Set styling on corrupted artifact figure
    ax_corr.set_title("Corruption of Artifacts Across Multiple Memory Dumps", pad=40)
    ax_corr.set_ylabel("Average Fraction of Artifact Intact")
    ax_corr.set_xlabel("Time Elapsed Since First Memory Dump (H:MM:SS)")
    legend = ax_corr.legend(loc="lower right")
    legend.set_title("Artifact Type")
    fig2.tight_layout(pad=3)

    # Set xtick labels if they are given.  Also rotate the xtick labels.
    if xtick_list:
        for ax in [ax_xpub, ax_rest, ax_corr]:
            ax.set_xticklabels(xtick_list, rotation=30)

def usage():
    print("Usage:\npython3 foreshadow-viz.py <csv_directory>\n")
    exit(1)

def main():

    # Make sure command line arguments are proper
    if len(sys.argv) != 2:
        usage()
    CSV_DIR = sys.argv[1]

    # Initialize Seaborn styling
    sns.set()

    # Load in all .csv files
    csv_names = get_csvs(CSV_DIR)
    raw_data_list = [pd.read_csv(osp.join(CSV_DIR, csv_name)) for csv_name in csv_names]

    # Original data - nothing to compare to
    data_list = [transform_data(raw_data_list[0], filename=csv_names[0])]
    # Compare all subsequent data to original / previous

    for index, data in enumerate(raw_data_list[1:]):
        prev = data_list[index]     # Index over this list is offset +1 from the original
        data_list += [transform_data(data, compare=True, original=prev, filename=csv_names[index+1])]

    # Get the number of artifacts by type for each memory dump
    print("Calculating counts of all artifacts.  ", end='')
    counts_all_list = [data.groupby("type")["start_addr_hex"].nunique()
                       for data in data_list]
    print("Done.\nCalculating counts of corrupted artifacts.  ", end='')
    counts_corr_list = [data[data["corruption_level"] < 1].groupby("type")["start_addr_hex"].nunique()
                        for data in data_list]
    print("Done.\nCalculating mean corruption level of all artifacts.  ", end='')
    means_corr_list = [data.groupby("type")["corruption_level"].mean()
                       for data in data_list]
    print("Done.")
    counts_data = [
        pd.DataFrame({
           "counts_all": counts_all,
           "counts_corrupted": counts_corrupted,
           "mean_corrupted": means_corrupted
        }).fillna(value=0)
        for counts_all, counts_corrupted, means_corrupted in zip(counts_all_list, counts_corr_list, means_corr_list)
    ]

    # Parse filename to extract timestamps and elapsed time
    times_list = [datetime.strptime(csv_name, "%Y-%m-%d--%H-%M-%S.csv")
                  for csv_name in csv_names]
    elapsed_list = [dump_time - times_list[0]
                    for index, dump_time in enumerate(times_list)]

    # Group data for plotting the differential analysis
    plot_differential_line_graph(data=counts_data, xtick_list=elapsed_list)

    # Plot memory artifacts
    # Uncomment this to print memory space frames
    """
    print("PLOTTING MEMORY SPACE" + "*"*80)
    for frame_num, frame in enumerate(data_list[:]):
        print(frame)
        plot_data(frame, frame_num=frame_num)
    plt.show()
    """

if __name__ == "__main__":
    main()
