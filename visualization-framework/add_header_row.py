"""
add_header_row.py
For each csv in a directory, add the proper header row if it does not have it.
This will allow the csv files to be processed by foreshadow-viz.py
"""
import os


def main():
    DIRECTORY = "csvs"
    EXTENSION = ".csv"
    BAD_HEADER_ROW = "type,id,address,size,data\n"
    HEADER_ROW = "type,id,start_addr_hex,size_bytes,data_base64\n"

    if not os.path.exists(DIRECTORY):
        print("\nERROR: Directory does not exist.\n")
    if not os.path.isdir(DIRECTORY):
        print("\nERROR: Path given is not a directory.\n")

    for filename in os.listdir(DIRECTORY):
        filepath = os.path.join(DIRECTORY, filename)

        # Skip non-csv files
        if not filename.endswith(EXTENSION):
            continue

        # Read current contents
        with open(filepath, "r") as file:
            data = file.readlines()

        # Skip if header row already there
        if data[0] in HEADER_ROW:
            print("Skipping", filename)
            continue
        if data[0] in BAD_HEADER_ROW:
            print("Replacing", filename)
            data[0] = HEADER_ROW
            data = ''.join(data)
        else:
            print("Writing ", filename)
            data = HEADER_ROW + ''.join(data)

        # Write data back to file with header row at beginning
        with open(filepath, "w") as file:
            file.write(data)


if __name__ == "__main__":
    main()
