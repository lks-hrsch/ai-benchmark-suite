import hashlib
import json
import os
import platform

import psutil
import torch

DEVICE_INFORMATION_FILENAME = "device_information.json"


class DeviceInformation:
    # properties
    # system information
    platform: str
    processor: str
    system: str
    release: str
    machine: str
    node: str
    # hardware information
    cpu_count: str
    memory: str
    cuda: str
    mps: str
    # python information
    python_version: str
    python_compiler: str
    python_implementation: str
    # torch information
    torch_version: str

    def __init__(self):
        """
        load device information from a json file
        """
        with open(DEVICE_INFORMATION_FILENAME, "r") as f:
            device_info = json.load(f)
            self.platform = device_info["platform"]
            self.processor = device_info["processor"]
            self.system = device_info["system"]
            self.release = device_info["release"]
            self.machine = device_info["machine"]
            self.node = device_info["node"]
            self.cpu_count = device_info["cpu_count"]
            self.memory = device_info["memory"]
            self.cuda = device_info["cuda"]
            self.mps = device_info["mps"]
            self.python_version = device_info["python_version"]
            self.python_compiler = device_info["python_compiler"]
            self.python_implementation = device_info["python_implementation"]
            self.torch_version = device_info["torch_version"]

    def to_dict(self):
        return {
            "device_name": self._get_device_hash(),
            "processor": self.processor,
            "system": self.system,
            "cpu_count": self.cpu_count,
            "memory": self.memory,
            "cuda": self.cuda,
            "mps": self.mps,
            "python_version": self.python_version,
            "python_compiler": self.python_compiler,
            "python_implementation": self.python_implementation,
            "torch_version": self.torch_version,
        }

    def _gather_device_information():
        device_info = {
            #
            # system information
            "platform": platform.platform(),
            "processor": platform.processor(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
            #
            # hardware information
            "cpu_count": os.cpu_count(),
            "memory": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            #
            # python information
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "python_implementation": platform.python_implementation(),
            #
            # torch information
            "torch_version": torch.__version__,
        }
        return device_info

    def _get_device_hash(self):
        """
        i want to anonymize the device information by hashing the platform, node, machine and release together
        """
        hash_input = f"{self.platform}{self.node}{self.machine}{self.release}".encode(
            "utf-8"
        )
        return hashlib.sha256(hash_input).hexdigest()

    @classmethod
    def save_device_information(cls, filename):
        device_info = cls._gather_device_information()
        with open(filename, "w") as f:
            json.dump(device_info, f)


if __name__ == "__main__":
    DeviceInformation.save_device_information(DEVICE_INFORMATION_FILENAME)
