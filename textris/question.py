import json
import pandas as pd
from typing import Dict, Optional, Union


class Question:
    # The question formatting code. Modified from Alex Lee's code.

    def __init__(
        self,
        question: str,
        answer: str,
        options: Dict[str, str],
        answer_idx: str,
        prompt_prefix: Optional[str] = "",
        prompt_suffix: Optional[str] = "",
    ):
        self.question = question
        self.answer = answer
        self.options = options
        self.answer_idx = answer_idx
        self.prompt_prefix = prompt_prefix if prompt_prefix else ""  # handle None
        self.prompt_suffix = prompt_suffix if prompt_suffix else ""  # handle None

        self._full_question = None

    def _get_full_question(
        self, prompt_prefix: Optional[str] = None, prompt_suffix: Optional[str] = None
    ) -> str:
        if prompt_prefix:
            self.prompt_prefix = prompt_prefix
        if prompt_suffix:
            self.prompt_suffix = prompt_suffix

        formatted_answers = "\n".join([f"{k}: {v}" for k, v in self.options.items()])

        # note that we are putting in a newline between the question and the answers
        full_question = f"""{self.prompt_prefix}{self.question} 

{formatted_answers}
{self.prompt_suffix}"""

        self._full_question = full_question

    def score(self, answer):
        if self.answer_idx in answer:
            return True
        else:
            return False

    @property
    def full_question(self):
        if self._full_question is None:
            self._get_full_question()
            return self._full_question
        else:
            return self._full_question

    @classmethod
    def from_series(self, series: Union[pd.Series, pd.DataFrame]):
        # TODO: add in keys of series as kwargs

        # if isinstance(element):

        return self(
            question=series["question"],
            answer=series["answer"],
            options=series["options"],
            answer_idx=series["answer_idx"],
            prompt_prefix=series["prompt_prefix"]
            if series.get("prompt_prefix", None)
            else "",
            prompt_suffix=series["prompt_suffix"]
            if series.get("prompt_suffix", None)
            else "",
        )

    @classmethod
    def from_dict(self, dict: Dict[str, Union[str, dict]]):
        return self(
            question=dict["question"],
            answer=dict["answer"],
            options=dict["options"],
            answer_idx=dict["answer_idx"],
            prompt_prefix=dict["prompt_prefix"]
            if dict.get("prompt_prefix", None)
            else "",
            prompt_suffix=dict["prompt_suffix"]
            if dict.get("prompt_suffix", None)
            else "",
        )