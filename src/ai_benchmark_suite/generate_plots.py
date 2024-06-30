"""
For each result file, generate a plot of the results.
We shoult create a plot for each device used in the benchmark.
"""

import matplotlib.pyplot as plt
import pandas as pd

from .save_information import RESULT_FOLDER


def generate_plot(csv_file, data_point_key, lower_is_better=False):
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

        # order the dataframe by the data_point_key from highest to lowest
        device_df = device_df.sort_values(data_point_key, ascending=not lower_is_better)

        # Create the plot
        plt.figure(
            figsize=(11.69, 8.27)
        )  # Set the figure size to A4 dimensions in inches

        # For horizontal bar chart, switch the axes by using plt.barh
        plt.barh(device_df["hardware_name"], device_df[data_point_key])

        plt.yticks(rotation=45)

        plt.xlabel(data_point_key)  # x-axis is the data_point_key

        plt.subplots_adjust(left=0.20, bottom=0.15)  # make the lables fit

        orientation = "lower" if lower_is_better else "higher"

        plt.title(f"{csv_file} {data_point_key} for {device} ({orientation} is better)")
        plt.savefig(f"{RESULT_FOLDER}/{csv_file}_{device}_{data_point_key}.png")


def main():
    # Example usage
    generate_plot("mnist", "average_training_time_per_epoch", True)
    generate_plot("qwen2-1_5B", "num_generated_tokens_per_second")
    generate_plot("qwen2-7B", "num_generated_tokens_per_second")


if __name__ == "__main__":
    main()
