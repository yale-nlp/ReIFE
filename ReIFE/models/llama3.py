from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
)
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
import torch
from .registry import register_model
import time
import threading
import warnings

class APIWarning(Warning):
    pass

warnings.simplefilter("once", APIWarning)  # show warning only once per


@register_model("llama3")
class Llama3(BaseLLM):
    def __init__(
        self,
        model_pt="meta-llama/Meta-Llama-3-8B-Instruct",
        device="auto",
        pad_token="<pad>",
    ):
        """
        Initializes the Llama3 model.

        Args:
            model_pt (str, optional): The path or name of the pre-trained model. Defaults to "meta-llama/Meta-Llama-3-8B-Instruct".
            device (str, optional): The device to run the model on. Defaults to "auto".
            pad_token (str, optional): The padding token. Defaults to "<pad>".
        """

        self.pad_token = pad_token
        self.device = device
        tokenizer = AutoTokenizer.from_pretrained(model_pt, use_fast=False)
        if self.device == "auto":
            model = LlamaForCausalLM.from_pretrained(
                model_pt, torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                model_pt, torch_dtype=torch.bfloat16
            ).to(device)
        # add pad token
        tokenizer.add_special_tokens({"pad_token": pad_token})
        embedding = model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        vocab_size = embedding.weight.shape[0]
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.embed_tokens.padding_idx = tokenizer.pad_token_id
        model.config.vocab_size = vocab_size
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.max_len = 8192

    def get_scoring_prompt(self, dialog):
        assert dialog[-1]["role"] == "assistant"  # last message should be assistant
        tokenized_example = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=False,
        )
        if len(tokenized_example) > self.max_len:
            print("Warning: input length exceeds max_len")
        input_ids = tokenized_example[: self.max_len]

        prompt_example = self.tokenizer.apply_chat_template(
            dialog[:-1],
            add_generation_prompt=True,
        )
        prompt_input_ids = prompt_example[: self.max_len]
        return input_ids, (len(prompt_input_ids), len(input_ids))


@register_model("llama3vllm")
class Llama3VLLM(BaseVLLM):
    """
    A class for the Llama3 VLLM model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.STOP_TOKEN_IDS = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens = self.tokenizer.apply_chat_template(
            dialog,
            add_generation_prompt=True,
        )
        if max_input_len is None:
            max_input_len = self.max_input_len
        if len(dialog_tokens) > max_input_len:
            print(
                f"Warning: input length {len(dialog_tokens)} exceeds max input length {max_input_len}"
            )
            dialog_tokens = dialog_tokens[:max_input_len]
        return dialog_tokens


@register_model("llama3.1vllm")
class Llama3_1VLLM(Llama3VLLM):
    """
    A class for the Llama3 VLLM model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.STOP_TOKEN_IDS = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|eom_id|>"),
        ]


@register_model("llama3.1api")
class Llama3_1VLLM(BaseLLMAPI):
    """
    A class for the Llama3 LLM API model.
    """

    def __init__(
        self,
        model_pt: str,
        key_path: str,
        account_path: str,
        parallel_size: int,
        max_retries: int = 10,
        initial_wait_time: int = 2,
        end_wait_time: int = 0,
        refresh_interval: int = 3500,
    ):
        """
        Initializes the BaseLLMAPI object for calling API services.

        Args:
            model_pt (str): Model name
            key_path (str): Path to the key file (should be kept secret)
            account_path (str): Path to the account file (should be kept secret)
            parallel_size (int): Number of parallel processes
            max_retries (int, optional): Maximum number of retries. Defaults to 10.
            initial_wait_time (int, optional): Initial wait time. Defaults to 2.
            end_wait_time (int, optional): End wait time. Defaults to 0.
            refresh_interval (int, optional): Refresh interval. Defaults to 3500.
        """

        from google.auth import default

        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

        # Authentication
        # credentials, _ = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        credentials, _ = default()
        self.credentials = credentials
        self.model_name = model_pt
        with open(account_path, "r") as f:
            organization = f.read().strip()
        self.organization = organization
        self.model_location = "us-central1"
        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time
        self.refresh_interval = refresh_interval

        self.refresh_token()
        self.refresh_thread = threading.Thread(target=self.refresh_token_periodically)
        self.refresh_thread.daemon = True
        self.refresh_thread.start()

    def refresh_token(self):
        """
        Refreshes the token.
        """
        from openai import OpenAI
        from google.auth.transport import requests

        auth_request = requests.Request()
        self.credentials.refresh(auth_request)
        self.client = OpenAI(
            api_key=self.credentials.token,
            base_url=f"https://{self.model_location}-aiplatform.googleapis.com/v1beta1/projects/{self.organization}/locations/{self.model_location}/endpoints/openapi/chat/completions?",
        )
        print("Token refreshed")

    def refresh_token_periodically(self):
        """
        Periodically refreshes the token.
        """
        while True:
            time.sleep(self.refresh_interval)
            self.refresh_token()

    def _get_response(
        self,
        prompt: list[dict],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
    ) -> list[dict]:
        """
        Get the response from the API service.

        Args:
            prompt (list[dict]): The prompt to be sent to the API service.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.

        Returns:
            list[dict]: The response from the API service. Each response is a dictionary containing the generated text ("text"), log probabilities ("logprobs", optional), and tokens ("tokens", optional).
        """
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        if n > 1:
            warnings.warn(
                f"Llama3.1API does not support multiple generations per prompt. Iterating over {n=:}.",
                APIWarning,
                stacklevel=2,
            )
        if logprobs is not None:
            warnings.warn(
                "Llama3.1API does not support log probabilities. Ignoring logprobs.",
                APIWarning,
                stacklevel=2,
            )
        all_choices = []
        for t in range(n):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response = response.model_dump()
            choices = response["choices"]
            for i in range(len(choices)):
                choices[i]["text"] = choices[i]["message"]["content"]
                choices[i]["logprobs"] = None
                choices[i]["tokens"] = None
            all_choices.extend(choices)
            if self.end_wait_time > 0 and t > 0:
                time.sleep(self.end_wait_time)
        return all_choices
