from .generate_plots import main as generate_plots_main
from .mnist.__main__ import main as mnist_main
from .qwen2.__main__ import main as qwen2_main

if __name__ == "__main__":
    mnist_main()
    qwen2_main()
    generate_plots_main()
