import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..device_information import DeviceInformation
from ..MyArgumentParser import MyArgumentParser
from ..save_information import safe_measure_point

example_prompts = [
    "Give me a short introduction to large language model.",
    "What is the most important thing in life?",
    "What is the best way to learn a new language?",
    "Where is the best place to go on vacation?",
    "What is the best way to learn a new skill?",
]


def load_model_and_tokenizer(MODEL_ID, device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tokenizer


def generate_model_inputs(device, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    return model_inputs


def statistics(device, inference_times, generated_tokens, file_name):
    print(
        f"Average inference time per example: {sum(inference_times) / len(inference_times)}"
    )
    print(
        f"Average number of generated tokens per example: {sum(generated_tokens) / len(generated_tokens)}"
    )
    print(
        "Number of generated Tokens per second: ",
        sum(generated_tokens) / sum(inference_times),
    )

    safe_measure_point(
        {
            "device": device,
            "average_inference_time": sum(inference_times) / len(inference_times),
            "total_inference_time": sum(inference_times),
            "average_generated_tokens": sum(generated_tokens) / len(generated_tokens),
            "num_generated_tokens_per_second": sum(generated_tokens)
            / sum(inference_times),
        },
        file_name,
    )


def _1_5B_Instruct(device):
    MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

    model, tokenizer = load_model_and_tokenizer(MODEL_ID, device)

    inference_times = []
    generated_tokens = []

    for prompt in example_prompts:
        model_inputs = generate_model_inputs(device, tokenizer, prompt)

        start_time = time.time()
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        end_time = time.time()

        num_generated_tokens = sum(len(ids) for ids in generated_ids)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

        generated_tokens.append(num_generated_tokens)

        inference_time = end_time - start_time
        inference_times.append(inference_time)
        print(f"Inference time: {inference_time}")

    file_name = "qwen2-1_5B.csv"

    del model
    del tokenizer

    statistics(device, inference_times, generated_tokens, file_name)


def _7B_Instruct(device):
    MODEL_ID = "Qwen/Qwen2-7B-Instruct"

    model, tokenizer = load_model_and_tokenizer(MODEL_ID, device)

    inference_times = []
    generated_tokens = []

    for prompt in example_prompts:
        model_inputs = generate_model_inputs(device, tokenizer, prompt)

        start_time = time.time()
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        end_time = time.time()

        num_generated_tokens = sum(len(ids) for ids in generated_ids)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)

        generated_tokens.append(num_generated_tokens)

        inference_time = end_time - start_time
        inference_times.append(inference_time)
        print(f"Inference time: {inference_time}")

    file_name = "qwen2-7B.csv"

    del model
    del tokenizer

    statistics(device, inference_times, generated_tokens, file_name)


def main():
    device_information = DeviceInformation()

    parser = MyArgumentParser(description="Qwen2-1.5B-Instruct Example")
    _ = parser.parse_args()
    device = parser.use_torch_device()

    print(device)

    _1_5B_Instruct(device)

    # if device RAM is less than 24GB then dont run the 7B model
    if int(device_information.memory) >= 24:
        _7B_Instruct(device)


if __name__ == "__main__":
    main()
