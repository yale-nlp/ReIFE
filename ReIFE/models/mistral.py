from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
import torch
from .registry import register_model
import warnings

class APIWarning(Warning):
    pass

warnings.simplefilter("once", APIWarning)  # show warning only once per

@register_model("mistral7bv1")
class Mistral7BV1(BaseLLM):
    def __init__(
        self,
        model_pt="mistralai/Mistral-7B-Instruct-v0.1",
        device="auto",
        pad_token="</s>",
        system_prompt: str | None = None,
    ):
        """
        Initializes the Mistral model.

        Args:
            model_pt (str, optional): The path or name of the pre-trained model. Defaults to "allenai/tulu-2-7b".
            device (str, optional): The device to run the model on. Defaults to "auto".
            pad_token (str, optional): The padding token. Defaults to "<pad>".
        """

        self.pad_token = pad_token
        self.device = device
        tokenizer = AutoTokenizer.from_pretrained(model_pt, use_fast=False)
        if self.device == "auto":
            model = AutoModelForCausalLM.from_pretrained(
                model_pt, torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_pt, torch_dtype=torch.bfloat16
            ).to(device)
        # add pad token
        tokenizer.add_special_tokens({"pad_token": pad_token})
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.model.embed_tokens.padding_idx = tokenizer.pad_token_id
        model.config.vocab_size = 32008
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.max_len = 32768

        self.B_INST = "[INST]"
        self.E_INST = "[/INST]"
        if system_prompt is not None:
            self.DEFAULT_SYSTEM_PROMPT = system_prompt
        else:
            self.DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def get_scoring_prompt(self, dialog):
        def _concat_messages(messages):
            if messages[0]["role"] != "system":
                messages = [
                    {
                        "role": "system",
                        "content": self.DEFAULT_SYSTEM_PROMPT,
                    }
                ] + messages
            
            messages[1]["content"] = messages[0]["content"].strip() + "\n" + messages[1]["content"]
            messages = messages[1:]
            assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
                [msg["role"] == "assistant" for msg in messages[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
            return dialog_tokens

        assert dialog[-1]["role"] == "assistant"  # last message should be assistant
        tokenized_example = _concat_messages(dialog).flatten()
        if len(tokenized_example) > self.max_len:
            print("Warning: input length exceeds max_len")
        input_ids = tokenized_example[: self.max_len]

        print(dialog[:-1])
        prompt_example = _concat_messages(dialog[:-1] + [{"role": "assistant", "content": ""}]).flatten()
        prompt_input_ids = prompt_example[: self.max_len]
        return input_ids, (len(prompt_input_ids), len(input_ids))

@register_model("mistral7bv1vllm")
class Mistral7BV1VLLM(BaseVLLM):
    """
    A class for the Llama2 VLLM model.
    """
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
    B_INST = "[INST]"
    E_INST = "[/INST]"

    def get_generation_prompt(self, dialog: list[dict]) -> torch.Tensor:
        if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": self.DEFAULT_SYSTEM_PROMPT,
                }
            ] + dialog
        dialog[1]["content"] = dialog[0]["content"].strip() + "\n" + dialog[1]["content"]
        dialog = dialog[1:]

        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens = self.tokenizer.apply_chat_template(dialog, return_tensors="pt")
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        if len(dialog_tokens) > self.max_input_len:
            print(f"Warning: input length {len(dialog_tokens)} exceeds max input length {self.max_input_len}")
            dialog_tokens = dialog_tokens[: self.max_input_len]
        return dialog_tokens


@register_model("mistralvllm")
class MistralVLLM(BaseVLLM):
    """
    A class for the Mistral VLLM model.
    """
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        if dialog[0]["role"] != "system":
            dialog = [
                {
                    "role": "system",
                    "content": self.DEFAULT_SYSTEM_PROMPT,
                }
            ] + dialog
        dialog[1]["content"] = dialog[0]["content"].strip() + "\n" + dialog[1]["content"]
        dialog = dialog[1:]
        
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
    

@register_model("mistralapi")
class MistralAPI(BaseLLMAPI):
    def __init__(
        self,
        model_pt: str,
        key_path: str,
        account_path: str,
        parallel_size: int,
        max_retries: int = 10,
        initial_wait_time: int = 2,
        end_wait_time: int = 0,
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
        """
        from openai import OpenAI

        self.model_name = model_pt
        with open(key_path, "r") as f:
            api_key = f.read().strip()

        self.client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time

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
                "MistralAPI does not support multiple generations per prompt. Using n=1.",
                APIWarning,
                stacklevel=2,
            )
        if logprobs is not None:
            warnings.warn(
                "MistralAPI does not support log probabilities. Ignoring logprobs.",
                APIWarning,
                stacklevel=2,
            )
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
        return choices