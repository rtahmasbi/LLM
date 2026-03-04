# env
```sh
conda create -n system_rca python=3.11
conda activate system_rca
pip install -r requirements.txt
```

# run
```sh
conda activate system_rca
cd /home/ras/GITHUB/LLM/AgenticAI/system_rca/

python main.py

```


# details
`run_diagnosis` is the main agent:
```py
from src.agent import run_diagnosis
```
and the graph is:
```
START -> analyst
analyst -> should_continue (conditioanl edge to "tool_executor" or "reporter")
tool_executor -> analyst
reporter -> END
```

