We can use https://www.agentql.com/, but it is a paid platform.
My goal is to create a simple version using `playwright`.




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

```

