import dotenv
import os
import argparse
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a lit of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()


if __name__ == "__main__":
    llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

    code_prompt = PromptTemplate(
        template="Write a very short {language} function that will {task}",
        input_variables=["language", "task"],
    )

    test_prompt = PromptTemplate(
        template="Write a test for the following {language} code:\n{code}",
        input_variables=["language", "code"],
    )

    code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")
    test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

    chain = SequentialChain(
        chains=[code_chain, test_chain],
        input_variables=["language", "task"],
        output_variables=["code", "test"],
    )

    results = chain({"language": args.language, "task": args.task})

    print(">>>>>> GENERATED CODE:")
    print(results["code"])

    print(">>>>>> GENERATED TEST:")
    print(results["test"])
