We can use https://www.agentql.com/, but it is a paid platform.
My goal is to create a simple version using `playwright`.

# examples
https://github.com/ajitsingh98/Auto-Job-Form-Filler-Agent
They fill Google Form


https://github.com/mohammed97ashraf/ApplyWizard
They use AgentQL. AgentQL is not free and all the LLM page extractions are there.




```sh
conda create -n agent_fill_online_form python=3.11

conda activate agent_fill_online_form

python -m pip install playwright
playwright install-deps
playwright install


pip install -U langchain
pip install -U langchain-openai
pip install -U langchain-anthropic
pip install -U langchain-community pypdf
pip install grandalf
pip install pygraphviz

```

