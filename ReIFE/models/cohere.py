from ..base_llm import BaseLLMAPI, BaseVLLM
from .registry import register_model
from transformers import AutoTokenizer
from vllm import LLM

@register_model("cohere_chat")
class CohereChat(BaseLLMAPI):
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
            end_wait_time (int, optional): End wait time. Defaults to 0.
        """
        super().__init__(
            model_pt,
            key_path,
            account_path,
            parallel_size,
            max_retries,
            initial_wait_time,
            end_wait_time,
        )
        import cohere
        self.co = cohere.Client(self.key)
        

    def _get_response(
        self,
        prompt: list[dict],
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.99,
        logprobs: int | None = None,
    ) -> list[dict]:
        """
        Get the response from the API service.

        Args:
            prompt (list[dict]): The prompt to be sent to the API service.
            n (int, optional): Number of text generations per prompt. Defaults to 1.
            max_tokens (int, optional): Maximum number of tokens in the generated text. Defaults to 1024.
            temperature (float, optional): Controls the randomness of the generated text. Higher values make the text more random. Defaults to 1.0.
            top_p (float, optional): Controls the diversity of the generated text. Lower values make the text more focused. Defaults to 0.99.
            logprobs (int | None, optional): Number of log probabilities to include in the generated text. Defaults to None.

        Returns:
            list[dict]: The response from the API service. Each response is a dictionary containing the generated text ("text"), log probabilities ("logprobs", optional), and tokens ("tokens", optional).
        """
        if n > 1:
            print(
                "Warning: Command R-Plus does not support multiple generations per prompt. Using n=1."
            )
        if logprobs is not None:
            print(
                "Warning: Command R-Plus does not support log probabilities. Ignoring logprobs."
            )
        if prompt[-1]["role"] != "user":
            raise ValueError("Last message should be user")
        if len([x for x in prompt if x["role"] == "user"]) > 1:
            raise ValueError("Only single round of conversation is supported")
        top_p = min(top_p, 0.99) # Command R-Plus only supports top_p up to 0.99
        prompt = prompt[-1]["content"]
        response = self.co.chat(
            message=prompt,
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            p=top_p,
        )
        response_text = response.text
        return [{"text": response_text}]
    

@register_model("coherevllm")
class CohereVLLM(BaseVLLM):
    """
    A class for the Cohere VLLM model.
    """
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
    ):
        """
        Initializes the Cohere object.

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

        tokenizer = AutoTokenizer.from_pretrained(model_pt, use_fast=True)
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
            tokenizer_mode="auto",
        )
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_model_len = max_model_len
    
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