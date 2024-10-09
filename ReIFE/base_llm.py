from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import vllm
from vllm import LLM, SamplingParams
from tqdm import tqdm
from abc import ABC, abstractmethod
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import time

vllm_version = vllm.__version__


class BaseLLM(ABC):
    def __init__(self, model_pt: str, device: str = "auto") -> None:
        """
        Initializes the BaseLLM class.

        Args:
            model_pt (str): The path to the pre-trained model.
            device (str, optional): The device to use for inference. Defaults to "auto".
        """

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_pt)
        if self.device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_pt, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_pt).to(device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            raise ValueError("Tokenizer does not have pad token")

    def left_pad(self, tensors: torch.Tensor, max_length: int = -1) -> torch.Tensor:
        """
        Left pads a batch of tensors with the specified maximum length.

        Args:
            tensors (torch.Tensor): The batch of tensors to be padded.
            max_length (int, optional): The maximum length to pad the tensors to. If not provided, the maximum length
                among the tensors in the batch will be used. Defaults to -1.

        Returns:
            torch.Tensor: The padded batch of tensors.
        """
        if max_length < 0:
            max_length = max([len(t) for t in tensors])
        padded_tensors = torch.full(
            (len(tensors), max_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        for i, t in enumerate(tensors):
            padded_tensors[i, -len(t) :] = torch.tensor(t, dtype=torch.long)
        return padded_tensors

    def right_pad(self, tensors: torch.Tensor, max_length: int = -1) -> torch.Tensor:
        """
        Right pads a batch of tensors with the specified maximum length.

        Args:
            tensors (torch.Tensor): The batch of tensors to be padded.
            max_length (int, optional): The maximum length to pad the tensors to. If not provided, the maximum length
                among the tensors in the batch will be used. Defaults to -1.

        Returns:
            torch.Tensor: The padded batch of tensors.
        """
        if max_length < 0:
            max_length = max([len(t) for t in tensors])
        padded_tensors = torch.full(
            (len(tensors), max_length), self.tokenizer.pad_token_id, dtype=torch.long
        )
        for i, t in enumerate(tensors):
            padded_tensors[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        return padded_tensors

    @abstractmethod
    def get_scoring_prompt(
        self, dialog: list[dict]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Gets the scoring prompt for the model.

        Args:
            dialog (list[dict]): The dialog to be scored.

        Returns:
            tuple[torch.Tensor, tuple[int, int]]: The scoring prompt and the indices of the assistant response.
        """
        pass

    def score(self, prompts: list[list[dict]]) -> list[list[float]]:
        """
        Scores a list of prompts (logs probabilities of the assistant response).

        Args:
            prompts (list[list[dict]]): The list of prompts to be scored.

        Returns:
            list[list[float]]: The scores for each prompt.
        """
        with torch.no_grad():
            inputs = [self.get_scoring_prompt(prompt) for prompt in prompts]
            input_ids = [x[0] for x in inputs]
            input_ids = self.right_pad(input_ids)
            response_idxs = [x[1] for x in inputs]
            if self.device != "auto":
                input_ids = input_ids.to(self.device)
            else:
                input_ids = input_ids.to("cuda")
            attention_mask = (input_ids != self.tokenizer.pad_token_id,)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_labels = shift_labels.to(shift_logits.device)
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            _log_probs = log_probs.cpu().numpy().tolist()
            log_probs = []
            for i in range(len(_log_probs)):
                start_idx, end_idx = response_idxs[i]
                assert start_idx < end_idx
                log_probs.append(_log_probs[i][start_idx:end_idx])
            return log_probs


class BaseVLLM(ABC):
    STOP_TOKEN_IDS = None

    def __init__(
        self,
        model_pt: str,
        tensor_parallel_size: int,
        max_input_len: int,
        max_model_len: int,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        dtype: str = "auto",
        quantization: str | None = None,
        download_dir: str | None = None,
        enforce_eager: bool = False,
        tokenizer_mode: str = "slow",
    ):
        """
        Initializes the BaseLLM object.

        Args:
            model_pt (str): The path to the pre-trained model.
            tensor_parallel_size (int): The size of the tensor parallelism.
            max_input_len (int): The maximum length of the input.
            max_model_len (int): The maximum length of the model.
            gpu_memory_utilization (float, optional): The GPU memory utilization. Defaults to 0.9.
            swap_space (int, optional): The swap space. Defaults to 4.
            dtype (str, optional): The data type. Defaults to "auto".
            quantization (str | None, optional): The quantization method. Defaults to None.
            download_dir (str | None, optional): The download directory. Defaults to None.
            enforce_eager (bool, optional): Whether to enforce eager execution. Defaults to False.
            tokenizer_mode (str, optional): The tokenizer mode. Defaults to "slow".
        """

        tokenizer = AutoTokenizer.from_pretrained(model_pt, use_fast=False, trust_remote_code=True)
        self.model = LLM(
            model=model_pt,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=download_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            swap_space=swap_space,
            quantization=quantization,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=True,
        )
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_model_len = max_model_len

    @abstractmethod
    def get_generation_prompt(
        self, dialog: list[dict], max_input_len: int | None = None
    ) -> list[int]:
        """
        Generates the input IDs for the language model based on the given dialog.

        Args:
            dialog (list[dict]): The dialog containing messages from different roles.
            max_input_len (int | None, optional): The maximum input length. If not provided, the default model input length will be used. Defaults to None.

        Returns:
            list[int]: The input IDs for the language model.
        """
        pass

    def generate(
        self,
        prompts: list[list[dict]],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
        use_tqdm: bool = True,
        input_length: int | None = None,
    ) -> list[list[dict]]:
        """
        Generates text based on the given prompts using the language model.

        Args:
            prompts (list[list[dict]]): List of prompts, where each prompt is a list of dictionaries.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.
            use_tqdm (bool, optional): Whether to display a progress bar. Defaults to True.
            input_length (int | None, optional): The length of the input. If not provided, the default model input length will be used. Defaults to None.

        Returns:
            list[list[dict]]: List of generated text, where each generated text is a list of dictionaries containing the generated text, log probabilities, and tokens.
        """

        max_input_len = self.max_input_len if input_length is None else input_length

        if max_tokens + max_input_len > self.max_model_len:
            raise ValueError(
                f"max_tokens ({max_tokens}) + max_input_len ({max_input_len}) > max_model_len ({self.max_model_len})"
            )

        prompts = [
            self.get_generation_prompt(prompt, input_length)
            for prompt in tqdm(prompts, desc="preparing prompts", disable=not use_tqdm)
        ]

        outputs = self.model.generate(
            prompt_token_ids=prompts,
            sampling_params=SamplingParams(
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                logprobs=logprobs,
                stop_token_ids=self.STOP_TOKEN_IDS,
            ),
            use_tqdm=use_tqdm,
        )
        _outputs = []
        for output in outputs:
            output = output.outputs
            _output = []
            for x in output:
                if logprobs is not None:
                    _logprobs = []
                    for y in x.logprobs:
                        if vllm_version >= "0.4.0":
                            _logprobs.append(
                                {self.tokenizer.decode(k): v.logprob for k, v in y.items()}
                            )
                        else:
                            _logprobs.append(
                                {self.tokenizer.decode(k): v for k, v in y.items()}
                            )
                else:
                    _logprobs = None
                tokens = [self.tokenizer.decode(y) for y in x.token_ids]
                text = x.text
                if self.STOP_TOKEN_IDS is not None:
                    for stop_token_id in self.STOP_TOKEN_IDS:
                        stop_token = self.tokenizer.decode(stop_token_id)
                        if text.endswith(stop_token):
                            text = text[: -len(stop_token)]
                            break
                _output.append(
                    {
                        "text": text,
                        "logprobs": _logprobs,
                        "tokens": tokens,
                    }
                )
            _outputs.append(_output)
        return _outputs


class BaseLLMAPI(ABC):
    def __init__(
        self,
        model_pt: str,
        key_path: str,
        account_path: str | None = None,
        parallel_size: int = 1,
        max_retries: int = 10,
        initial_wait_time: int = 2,
        end_wait_time: int = 0,
    ):
        """
        Initializes the BaseLLMAPI object for calling API services.

        Args:
            model_pt (str): Model name
            key_path (str): Path to the key file (should be kept secret)
            account_path (str | None, optional): Path to the account file (should be kept secret). Defaults to None.
            parallel_size (int, optional): Number of parallel processes. Defaults to 1.
            max_retries (int, optional): Maximum number of retries. Defaults to 10.
            initial_wait_time (int, optional): Initial wait time. Defaults to 2.
            end_wait_time (int, optional): The wait time after the successful request. Defaults to 0.
        """

        self.model_name = model_pt

        if key_path :
            with open(key_path, "r") as f:
                self.key = f.read().strip()
        if account_path:
            with open(account_path, "r") as f:
                self.account = f.read().strip()

        self.parallel_size = parallel_size
        self.max_retries = max_retries
        self.initial_wait_time = initial_wait_time
        self.end_wait_time = end_wait_time

    @abstractmethod
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
        pass

    def get_response(
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
        retries = 0
        wait_time = self.initial_wait_time
        while retries < self.max_retries:
            try:
                response = self._get_response(
                    prompt=prompt,
                    n=n,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    logprobs=logprobs,
                )
                if self.end_wait_time > 0:
                    time.sleep(self.end_wait_time)
                return response
            except Exception as e:
                print(e)
                if retries == self.max_retries - 1:
                    # raise e
                    print("!!!WARNING: Max retries reached, returning empty response")
                    response = [{"text": ""}]
                    return response
                print("retrying...", retries, "sleep...", wait_time)
                time.sleep(wait_time)
                retries += 1
                wait_time *= 2

    def generate(
        self,
        prompts: list[list[dict]],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        logprobs: int | None = None,
        use_tqdm: bool = True,
        input_length: int | None = None,
    ) -> list[list[dict]]:
        """
        Generates text based on the given prompts using the language model.

        Args:
            prompts (list[list[dict]]): List of prompts, where each prompt is a list of dictionaries.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 1.0.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.
            use_tqdm (bool, optional): Whether to display a progress bar. Defaults to True.
            input_length (int | None, optional): The length of the input. If not provided, the default model input length will be used. Defaults to None. It is not used in the API version.

        Returns:
            list[list[dict]]: List of generated text, where each generated text is a list of dictionaries containing the generated text, log probabilities, and tokens.
        """
        if input_length is not None:
            print("Warning: input_length is not used in the API version")
        get_response_fn = partial(
            self.get_response,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        with ThreadPool(self.parallel_size) as pool:
            outputs = list(
                tqdm(
                    pool.imap(get_response_fn, prompts),
                    total=len(prompts),
                    desc="Generating",
                    disable=not use_tqdm,
                )
            )
        _outputs = []
        for output in outputs:
            _output = []
            for x in output:
                _output.append(
                    {
                        "text": x["text"],
                        "logprobs": x["logprobs"] if "logprobs" in x else None,
                        "tokens": x["tokens"] if "tokens" in x else None,
                        "usage": x["usage"] if "usage" in x else None,
                    }
                )
            _outputs.append(_output)
        return _outputs
