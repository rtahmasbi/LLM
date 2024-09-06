https://python.langchain.com/v0.1/docs/use_cases/chatbots/memory_management/

The main libraries are:

```py
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.memory import ChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.runnables import RunnablePassthrough


```

