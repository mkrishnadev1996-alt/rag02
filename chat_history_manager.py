from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
import tiktoken
import os
from dotenv import load_dotenv
load_dotenv()

# Class to manage chat history with token limit
# Inherets the BaseChatMessageHistory
# Uses model gpt-oss-20b
# Default Token limit is set to 2000 tokens
class TokenLimitedChatHistory(BaseChatMessageHistory):
    def __init__(self, max_tokens: int = 2000, model: str = "gpt-oss-20b"):
        self.messages: list[BaseMessage] = []
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
    
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self._trim_to_token_limit()
    
    def _trim_to_token_limit(self) -> None:
        while len(self.messages) > 1:
            total_tokens = sum(len(self.encoding.encode(msg.content)) for msg in self.messages)
            if total_tokens <= self.max_tokens:
                break
            self.messages.pop(0)
    
    def clear(self) -> None:
        self.messages = []

