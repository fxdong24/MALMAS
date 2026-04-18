""" Class for sampling new programs. """
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from . import evaluator
from . import buffer
from . import config as config_lib
import requests
import json
import http.client
import os



class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """ Return a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """ Return multiple predicted continuations of `prompt`. """
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]



class Sampler:
    """ Node that samples program skeleton continuations and sends them for analysis. """
    _global_samples_nums: int = 1

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            meta_data: dict,
            config: config_lib.Config,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM,
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._meta_data = meta_data
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums
        self.config = config
        self.__class__._global_samples_nums = 1

    
    def sample(self, **kwargs):
        """ Continuously gets prompts, samples programs, sends them for analysis. """
        while True:

            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break

            prompt = self._database.get_prompt()
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code,self.config)
            sample_time = (time.time() - reset_time) / self._samples_per_prompt
            # This loop can be executed in parallel on remote evaluator machines.
            for sample in samples:
                sample = "\n    import pandas as pd\n    import numpy as np\n" + sample
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.data_input,
                    prompt.data_output,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1



def _extract_body(sample: str, config: config_lib.Config) -> str:
    """
    Extract the function body from a response sample, removing any preceding descriptions
    and the function signature. Preserves indentation.
    ------------------------------------------------------------------------------------------------------------------
    Input example:
    ```
    This is a description...
    def function_name(...):
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    Output example:
    ```
        return ...
    Additional comments...
    ```
    ------------------------------------------------------------------------------------------------------------------
    If no function definition is found, returns the original sample.
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    
    for lineno, line in enumerate(lines):
        # find the first 'def' program statement in the response
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    
    if find_def_declaration:
        # for gpt APIs
        if config.use_api:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
        
        # for mixtral
        else:
            code = ''
            indent = '    '
            for line in lines[func_body_lineno + 1:]:
                if line[:4] != indent:
                    line = indent + line
                code += line + '\n'
        
        return code
    
    return sample



class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, batch_inference: bool = True, trim=True) -> None:
        """
        Args:
            batch_inference: Use batch inference when sample equation program skeletons. The batch size equals to the samples_per_prompt.
        """
        super().__init__(samples_per_prompt)

        url = "http://127.0.0.1:5000/completions"
        instruction_prompt = ("You are a helpful assistant tasked with discovering new features/ dropping less important feaures for the given prediction task. \
                             Complete the 'modify_features' function below, considering the physical meaning and relationships of inputs.\n\n")
        self._batch_inference = batch_inference
        self._url = url
        self._instruction_prompt = instruction_prompt
        self._trim = trim


    def draw_samples(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        """Returns multiple equation program skeleton hypotheses for the given `prompt`."""
        if config.use_api:
            return self._draw_samples_api(prompt, config)
        else:
            return self._draw_samples_local(prompt, config)


    def _draw_samples_local(self, prompt: str, config: config_lib.Config) -> Collection[str]:    
        # instruction
        prompt = '\n'.join([self._instruction_prompt, prompt])
        while True:
            try:
                all_samples = []
                # response from llm server
                if self._batch_inference:
                    response = self._do_request(prompt)
                    for res in response:
                        all_samples.append(res)
                else:
                    for _ in range(self._samples_per_prompt):
                        response = self._do_request(prompt)
                        all_samples.append(response)

                # trim equation program skeleton body from samples
                if self._trim:
                    all_samples = [_extract_body(sample, config) for sample in all_samples]
                
                return all_samples
            except Exception:
                continue

    def _draw_samples_api(self, prompt: str, config: config_lib.Config) -> Collection[str]:
        all_samples = []
        prompt = '\n'.join([self._instruction_prompt, prompt])

        for _ in range(self._samples_per_prompt):
            while True:
                try:
                    if config.api_model.startswith("deep"):
                        conn = http.client.HTTPSConnection("api.deepseek.com")
                    elif config.api_model.startswith("gpt"):
                        conn = http.client.HTTPSConnection("api.chatanywhere.tech")
                    else:
                        print("model error")
                    payload = json.dumps({
                        "max_tokens": 512,
                        "model": config.api_model,  # e.g., "deepseek-chat"
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "temperature": 0.7
                    })
                    headers = {
                        'Authorization': f"Bearer {config.api_key}",  # << 显式填写你的 API Key
                        'User-Agent': 'PythonClient/1.0',
                        'Content-Type': 'application/json'
                    }
                    conn.request("POST", "/chat/completions", payload, headers)
                    res = conn.getresponse()
                    data = json.loads(res.read().decode("utf-8"))
                    response = data['choices'][0]['message']['content']

                    if self._trim:
                        response = _extract_body(response, config)

                    all_samples.append(response)
                    break

                except Exception:
                    continue

        return all_samples

    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        # repeat the prompt for batch inference
        repeat_prompt: int = self._samples_per_prompt if self._batch_inference else 1
        
        data = {
            'prompt': content,
            'repeat_prompt': repeat_prompt,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers)
        
        if response.status_code == 200: #Server status code 200 indicates successful HTTP request! 
            response = response.json()["content"]
            
            return response if self._batch_inference else response[0]

