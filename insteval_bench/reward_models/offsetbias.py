from ..base_rm import BaseRMHF
from .registry import register_model
from transformers import AutoTokenizer, pipeline
import torch

@register_model("offsetbias")
class OffsetBias(BaseRMHF):
    def __init__(
        self, model_pt: str, batch_size: int = 1, device: str = "cuda"
    ) -> None:
        """
        Initializes the OffsetBias RM object for calling Hugging Face pipelines.
        """
        super().__init__(model_pt, batch_size, device)
        rm_tokenizer = AutoTokenizer.from_pretrained(model_pt)
        self.rm_pipe = pipeline(
            "sentiment-analysis",
            model=model_pt,
            device=device,
            tokenizer=rm_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )
        self.tokenizer = rm_tokenizer

    @staticmethod
    def make_input(input: dict) -> dict:
        messages = [
            {"role": "user", "content": input["instruction"]},
            {"role": "assistant", "content": input["output"]},
        ]
        return messages
    
    def score_batch(self, inputs: list[dict], **kwargs) -> list[dict]:
        """
        Scores the inputs.

        Args:
            inputs (list[dict]): List of inputs.

        Returns:
            list[dict]: List of dictionaries containing the results
        """
        pipe_kwargs = {
            "return_all_scores": True,
            "function_to_apply": "none",
            "batch_size": len(inputs),
        }

        inputs = [self.make_input(input) for input in inputs]
        test_texts = [self.tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=False).replace(self.tokenizer.bos_token, "") for x in inputs]
        pipe_outputs = self.rm_pipe(test_texts, **pipe_kwargs)
        results = [{"score": x[0]["score"]} for x in pipe_outputs]
        return results