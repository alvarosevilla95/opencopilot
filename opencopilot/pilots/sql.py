from typing import Any, Dict
from IPython.display import clear_output
from langchain import LLMChain, PromptTemplate, SQLDatabase, SQLDatabaseChain
from langchain.schema import BaseLanguageModel

from opencopilot.notebook.utils import create_new_cell

db_query_prompt_template = """Given an input question, generate a correct {dialect} query to run, which will be run and loaded in a dataframe to be used to answer the question. Don't use limit in your queries, unless the question requires it.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question, up to {top_k}.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"

The SQL query should be outputted plainly, do not surround it in quote or else.
Be careful with ambiguous column names, particularly on the bid column. Always access it using the table.bid notation.

Only use the tables listed below.

{table_info}

Question: {input}"""

db_direct_query_prompt = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "top_k"],
    template=db_query_prompt_template,
)


def sql_query_cell(query: str, text_input: str, name: str):
    return f"""
query = \"\"\"
{query}
\"\"\"
{name} = pd.read_sql_query(query, conn)
{name}_description = "{text_input}"
datasets["{name}"] = ({name}, {name}_description)
""".strip()

class REPLDatabaseChain(SQLDatabaseChain):

    @classmethod
    def from_llm_and_db(cls, 
                        llm: BaseLanguageModel, 
                        db: SQLDatabase, 
                        verbose=False, 
                        return_direct=False, 
                        prompt=db_direct_query_prompt
                        ):

        return cls(llm=llm, database=db, verbose=verbose, return_direct=return_direct, prompt=prompt)

    def run(self, query):
        try:
            return self({self.input_key: query})
        except Exception as e:
            return str(e)
    
    def _get_result_as_cell(self, query, text_input):
        print(query)
        name = input("Choose df name: ")
        clear_output()
        cell = sql_query_cell(query, text_input, name)
        create_new_cell(cell)

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        input_text = f"{inputs[self.input_key]} \nSQLQuery:"
        self.callback_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        llm_inputs = {
            "input": input_text,
            "top_k": self.top_k,
            "dialect": self.database.dialect,
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }
        intermediate_steps = []
        sql_cmd = llm_chain.predict(**llm_inputs)
        intermediate_steps.append(sql_cmd)
        self.callback_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
        result = self._get_result_as_cell(sql_cmd, inputs[self.input_key])
        intermediate_steps.append(result)
        return {self.output_key: "success"}
