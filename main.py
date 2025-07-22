from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
#from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

def gen_pet_name(animal_type):
    llm = OpenAI(temperature=0.7, max_tokens=64, model="gpt-4.1-nano")
    prompt= PromptTemplate(
        input_variables=['animal_type'],
        template="Suggest five cute {animal_type} pet names."
    )
    name_chain = prompt | llm

    response= name_chain.invoke({'animal_type': animal_type})
    return response

if __name__ == "__main__":
    print(gen_pet_name("cow"))
