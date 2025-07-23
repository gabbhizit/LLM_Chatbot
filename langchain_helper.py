from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools

#from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()


load_dotenv()

def gen_pet_name(animal_type):
    llm = OpenAI(temperature=0.7, max_tokens=64)
    prompt= PromptTemplate(
        input_variables=['animal_type'],
        template="Suggest five cute {animal_type} pet names."
    )
    name_chain = prompt | llm

    response= name_chain.invoke({'animal_type': animal_type})
    return response

def langchain_agent():
    llm = OpenAI(temperature=0.7, max_tokens=64)
    tools = load_tools(["wikipedia","llm-math"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    result = agent.invoke("What is the average weight of a cow? Multiply that by 3")
    return result

if __name__ == "__main__":
    print(langchain_agent())
    #print(gen_pet_name("cow"))
