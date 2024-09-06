# cat > flask_app.py

from flask import Flask, make_response

app = Flask(__name__)
app.config["DEBUG"] = True


from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain  --> depricated

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(
    model_path="Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_0.gguf",
    n_gpu_layers=40,
    n_batch=512,  # Batch size for model processing
    verbose=False,  # Enable detailed logging for debugging
)

# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain to manage interactions with the prompt and model
# llm_chain = LLMChain(prompt=prompt, llm=llm) --> depricated

chain = prompt | llm


@app.route('/', methods=['GET'])
def home():
    response = make_response("chatbot is ready!", 200)
    response.mimetype = "text/plain"
    return response



@app.route('/<question>')
def questionchat_llm(question):
    answer = chain.invoke({"question": question})
    response = make_response(question + "\n\n" + answer, 200)
    response.mimetype = "text/plain"
    return response



#app.run()
app.run(host="0.0.0.0", port=9453, debug=False)

# it just uses 4GB in CPU RAM
