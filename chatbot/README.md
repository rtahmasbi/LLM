
https://medium.com/@weidagang/hello-llm-building-a-local-chatbot-with-langchain-and-llama2-3a4449fc4c03

(pdf)

# Create a project dir
```sh
mkdir llm_chatbot
cd llm_chatbot
mkdir models

# Initialize a virtualenv
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install langchain llama-cpp-python

```


# download Llama-2-7B-Chat-GGUF
```sh
git lfs install 
git clone https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
```

Note:
> k_s models for whatever reason are a little slower than k_m models. k models are k-quant models and generally have less perplexity loss relative to size. A q4_K_M model will have much less perplexity loss than a q4_0 or even a q4_1 model.

https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md




```py
# cat > llm_chatbot.py

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

print("Chatbot initialized, ready to chat...")
while True:
    question = input("> ")
    #answer = llm_chain.run(question)
    answer = chain.invoke({"question": question})
    print(answer)


```


`llama-2-7b-chat.Q4_0.gguf` just uses 4GB in CPU RAM

The speed is reasonable running on CPU!!!
