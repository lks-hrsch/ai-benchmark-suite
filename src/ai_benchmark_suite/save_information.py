from datetime import datetime

from .device_information import DeviceInformation

RESULT_FOLDER = "results"


def safe_measure_point(data: dict, filename: str):
    device_information = DeviceInformation()

    data_point = device_information.to_dict()
    data_point.update(data)

    # also add the current date to the data_point
    data_point["date"] = datetime.now().strftime("%Y-%m-%d")

    # if there is already a saved combination of device and device_name than we overwrite the existing data_point
    # TODO: this should be implemented

    # save data_point as .csv file at filename if the file does not exist set the keys as the header
    with open(f"{RESULT_FOLDER}/{filename}", "a") as f:
        if f.tell() == 0:
            f.write(",".join(data_point.keys()) + "\n")
        f.write(",".join([str(value) for value in data_point.values()]) + "\n")


if __name__ == "__main__":
    mock_data = {
        "test": "test",
    }
    mock_file = "mock_file.csv"
    safe_measure_point(mock_data, mock_file)
