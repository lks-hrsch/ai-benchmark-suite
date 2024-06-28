import torch
from argparse import ArgumentParser, Namespace


class MyArgumentParser(ArgumentParser):
    parsed_args: Namespace | None = None

    def __init__(self, description: str):
        super().__init__(description=description)

        # add basic arguments
        self.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
        self.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
        self.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')
        self.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
        self.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
        self.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

    def parse_args(self, *args, **kwargs) -> Namespace:
        self.parsed_args = super().parse_args(*args, **kwargs)
        return self.parsed_args
    
    def use_torch_device(self) -> torch.device:
        use_cuda = not self.parsed_args.no_cuda and torch.cuda.is_available()
        use_mps = not self.parsed_args.no_mps and torch.backends.mps.is_available()

        if use_cuda:
            return torch.device("cuda")  # nvidia gpu
        elif use_mps:
            return torch.device("mps") # macos gpu
        else:
            return torch.device("cpu") # cpu