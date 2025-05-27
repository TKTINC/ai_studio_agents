"""
Base cognition module for AI Studio Agents.

This module handles decision-making using the Claude LLM, including
processing inputs, generating responses, and making decisions.
"""

import anthropic
from typing import Any, Dict, List, Optional


class BaseCognitionModule:
    """
    Base cognition module for AI Studio Agents.
    
    Handles decision-making using the Claude LLM, including processing inputs,
    generating responses, and making decisions.
    """
    
    def __init__(self, llm_settings):
        """
        Initialize the cognition module.
        
        Args:
            llm_settings: Configuration for the LLM
        """
        self.llm_settings = llm_settings
        self.client = anthropic.Anthropic(api_key=llm_settings.api_key)
        self.system_prompt = llm_settings.system_prompt or self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for the agent.
        
        Returns:
            Default system prompt
        """
        return """
        You are an AI Assistant designed to help users with various tasks.
        
        Your primary goals are:
        1. Understand user requests accurately
        2. Provide helpful and informative responses
        3. Complete tasks efficiently
        4. Learn from interactions to improve over time
        
        You should be:
        - Clear and concise in your communications
        - Transparent about your capabilities and limitations
        - Responsive to user feedback
        - Respectful of user privacy and preferences
        
        You have access to various tools that will be provided through function calling.
        Always use the appropriate tool for each task.
        """
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set a custom system prompt.
        
        Args:
            prompt: The new system prompt
        """
        self.system_prompt = prompt
    
    def _format_messages(self, context: Dict[str, Any], input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format the conversation history and current input into messages for the LLM.
        
        Args:
            context: The current context from working memory
            input_data: The current input data
            
        Returns:
            Formatted messages for the LLM
        """
        messages = []
        
        # Add conversation history
        for entry in context.get("conversation", []):
            if "input" in entry and isinstance(entry["input"], dict) and "content" in entry["input"]:
                messages.append({"role": "user", "content": entry["input"]["content"]})
            if "response" in entry and isinstance(entry["response"], dict) and "content" in entry["response"]:
                messages.append({"role": "assistant", "content": entry["response"]["content"]})
        
        # Add current input
        if "content" in input_data:
            messages.append({"role": "user", "content": input_data["content"]})
        
        return messages
    
    async def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and generate a response using the LLM.
        
        Args:
            input_data: The input data to process
            context: The current context from working memory
            
        Returns:
            The LLM's response
        """
        messages = self._format_messages(context, input_data)
        
        response = await self.client.messages.create(
            model=self.llm_settings.model,
            system=self.system_prompt,
            messages=messages,
            max_tokens=self.llm_settings.max_tokens,
            temperature=self.llm_settings.temperature
        )
        
        return {
            "type": "text",
            "content": response.content[0].text,
            "metadata": {
                "model": self.llm_settings.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        }
