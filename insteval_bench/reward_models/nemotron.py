from ..base_rm import BaseRMAPI
from .registry import register_model

@register_model("nemotron")
class Nemotron(BaseRMAPI):
    def __init__(
        self,
        model_pt: str,
        key_path: str,
        account_path: str | None = None,
        parallel_size: int = 1,
        max_retries: int = 10,
        initial_wait_time: int = 2,
        end_wait_time: int = 0,
    ) -> None:
        """
        Initializes the BaseLLMAPI object for calling API services.

        Args:
            model_pt (str): Model name
            key_path (str): Path to the key file (should be kept secret)
            account_path (str | None, optional): Path to the account file (should be kept secret). Defaults to None.
            parallel_size (int): Number of parallel processes. Defaults to 1.
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
        self.WEIGHTS = {
            "helpfulness": 0.3,
            "correctness": 0.74,
            "coherence": 0.46,
            "complexity": 0.47,
            "verbosity": -0.33,
        }

    def _get_response(
        self,
        input: dict,
        verbose: bool = False,
    ) -> dict:
        """
        Get the response from the API service.

        Args:
            input (dict): The input to be sent to the API service.
            verbose (bool, optional): Whether to print the progress. Defaults to False.

        Returns:
            dict: The response from the API service. It should at least contain the assigned score.
        """
        messages = [
            {"role": "user", "content": input["instruction"]},
            {"role": "assistant", "content": input["output"]},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        response = response.model_dump()
        choice = response["choices"][0]
        raw_output = choice["message"][0]["content"]
        scores = {x["token"]: x["logprob"] for x in choice["logprobs"]["content"]}
        score = sum([scores[x] * self.WEIGHTS[x] for x in self.WEIGHTS])
        return {"score": score, "raw_output": raw_output, "scores": scores}