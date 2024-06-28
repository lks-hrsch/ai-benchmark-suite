"""
For each result file, generate a plot of the results.
We shoult create a plot for each device used in the benchmark.
"""

import matplotlib.pyplot as plt
import pandas as pd

from .save_information import RESULT_FOLDER


def generate_plot(csv_file, data_point_key):
    """
    Generate a plot for the given csv file and data point key.
    """

    # Read the csv file
    df = pd.read_csv(f"{RESULT_FOLDER}/{csv_file}.csv")

    # Get the unique device names
    devices = df["device"].unique()

    # Create a plot for each devicepoetry add
    for device in devices:
        # Get the data for the current device
        device_df = df[df["device"] == device]

        # Create the plot
        plt.figure()

        # For horizontal bar chart, switch the axes by using plt.barh
        plt.barh(device_df["device_name"], device_df[data_point_key])

        plt.ylabel("device_name")  # Now y-axis is device_name
        plt.xlabel(data_point_key)  # x-axis is the data_point_key

        plt.title(f"{csv_file} {data_point_key} for {device}")
        plt.savefig(f"{RESULT_FOLDER}/{csv_file}_{device}_{data_point_key}.png")


def main():
    # Example usage
    generate_plot("mnist", "average_training_time_per_epoch")


if __name__ == "__main__":
    main()
