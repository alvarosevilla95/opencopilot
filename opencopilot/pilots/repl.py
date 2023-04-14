import re
import pandas as pd
from typing import Dict, Tuple
from langchain import LLMChain, PromptTemplate
from langchain.schema import BaseLanguageModel
from pydantic import Field

from opencopilot.notebook.utils import create_new_cell

datasets = {}

repl_template = """You are an assistant on a jupter notebook using pandas, matplotlib and pyvis.
Your task is to generate a new cell based on the user's request.
Your response should be the cell body, nothing else. Do not surround it in snippet quotes or anything else.
The first thing in the cell will be a triple quoted string with a comment.
You have access to a series of dataframes that have been already loaded. They will be listed below.
{dfs}

The user has requested the following:
{input}

New cell:
"""

repl_prompt = PromptTemplate(
        template=repl_template,
        input_variables=["input", "dfs"],
        )

class REPLChain(LLMChain):

    dfs: Dict[str, Tuple[pd.DataFrame, str]] = Field(...)

    @classmethod
    def from_llm_and_datasets(cls, 
                              llm: BaseLanguageModel, 
                              datasets: Dict[str, Tuple[pd.DataFrame, str]] = datasets,
                              prompt=repl_prompt,
                              verbose=False) -> "REPLChain":
        return cls(llm=llm, dfs=datasets ,prompt=prompt, verbose=verbose)
    
    def run(self, input: str, dfs: Dict[str, Tuple[pd.DataFrame, str]] = datasets):
        df_strings = "\n".join([f"{name}: Columns: {df.columns} Description: '{desc}'." for name, (df, desc) in dfs.items()])
        return create_new_cell(self({"input": input, "dfs": df_strings})[self.output_key])


