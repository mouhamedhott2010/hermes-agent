"""
MiniMax M2.5 Tool Call Parser

MiniMax M2.5 uses XML-style tool calling format:
<minimax:tool_call>
<invoke name="function_name">
<parameter name="param1">value1</parameter>
</invoke>
</minimax:tool_call>
"""

import re
import json
import uuid
from typing import Dict, List, Optional, Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from environments.tool_call_parsers import ToolCallParser, register_parser, ParseResult


@register_parser("minimax_m25")
class MiniMaxM25ToolCallParser(ToolCallParser):
    """Parser for MiniMax M2.5 XML-style tool calls."""

    def parse(self, text: str) -> ParseResult:
        if "<minimax:tool_call>" not in text:
            return (text, None)

        tool_calls: List[ChatCompletionMessageToolCall] = []

        try:
            tool_call_regex = re.compile(
                r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
            )
            invoke_regex = re.compile(r'<invoke name="([^"]*)"(.*?)</invoke>', re.DOTALL)
            parameter_regex = re.compile(
                r'<parameter name="([^"]*)">(.*?)</parameter>', re.DOTALL
            )

            for tool_call_match in tool_call_regex.findall(text):
                for invoke_match in invoke_regex.findall(tool_call_match):
                    function_name = invoke_match[0]
                    invoke_content = invoke_match[1]

                    arguments_dict: Dict[str, Any] = {}
                    for param_match in parameter_regex.findall(invoke_content):
                        param_name = param_match[0]
                        param_value = param_match[1].strip()

                        try:
                            arguments_dict[param_name] = json.loads(param_value)
                        except json.JSONDecodeError:
                            arguments_dict[param_name] = param_value

                    tool_call = ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        type="function",
                        function={
                            "name": function_name,
                            "arguments": json.dumps(arguments_dict)
                        }
                    )
                    tool_calls.append(tool_call)

        except Exception:
            return (text, None)

        content = tool_call_regex.sub("", text).strip() or None
        return (content, tool_calls if tool_calls else None)


@register_parser("minimax")
class MiniMaxToolCallParser(MiniMaxM25ToolCallParser):
    pass
