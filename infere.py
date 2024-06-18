from llama_cpp import Llama
from typing import List
INSTRUCT = {"role": "system",
                 "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."}

class Inference:
    """Interface to infere a model"""
    def __init__(
        self,
        model_path,
        ) -> None:
        self.llm = Llama(
            model_path=model_path,
            flash_attn=True,
            n_gpu_layers=-1,
            n_ctx=8000,
            n_threads=4,
            use_mlock=False,
            verbose=False,
            # draft_model=LlamaPromptLookupDecoding(
            #     max_ngram_size=3, num_pred_tokens=5
            # ),  # boost?
        )
    
    def generate_response(
        self,
        prompts: List[str] | str,
    ) -> List[str]:
        """Method to generate response from the model.

        :param prompts: The prompts that will be used to generate the response
        :type prompts: List[str] | str
        :return: the generated response that corresponds to the prompt
        :rtype: List[str]
        """
        msgs = []
        if isinstance(prompts, list):
            for prompt in prompts:
                msgs.append(
                    self.llm.create_chat_completion(
                        [INSTRUCT, {"role": "user", "content": prompt}]
                    )["choices"][0]["message"]["content"]
                )
        else:
            msgs.append(
                self.llm.create_chat_completion(
                    [INSTRUCT, {"role": "user", "content": prompts}]
                )["choices"][0]["message"]["content"]
            )
        return msgs
