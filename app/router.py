from app.config import settings
import re
import logging

logger = logging.getLogger(__name__)

class PromptRouter:
    def __init__(self, complexity_threshold_chars: int = settings.COMPLEXITY_THRESHOLD_CHARS):
        self.complexity_threshold_chars = complexity_threshold_chars
        # Case-insensitive compile of keywords suggesting complex reasoning
        self.complex_keywords = re.compile(
            r"\b(solve|calculate|prove|debug|design|architect|optimize|analyze|explain|compare|contrast|summarize|reason)\b",
            re.IGNORECASE
        )

    def is_complex(self, prompt: str) -> bool:
        """
        Classifies a prompt as simple or complex based on length and key complexity markers.
        Returns:
            True if the prompt requires cloud LLM reasoning, False if it can be handled locally.
        """
        # Rule 1: Empty or extremely short prompts are handled locally
        if not prompt or len(prompt.strip()) < 10:
            logger.info("Routing: Prompt is extremely short. Classifying as 'simple'.")
            return False

        # Rule 2: Prompt length threshold check
        prompt_len = len(prompt)
        if prompt_len > self.complexity_threshold_chars:
            logger.info(f"Routing: Prompt length ({prompt_len} chars) exceeds threshold ({self.complexity_threshold_chars} chars). Classifying as 'complex'.")
            return True

        # Rule 3: Keyword search for complex cognitive tasks
        if self.complex_keywords.search(prompt):
            logger.info("Routing: Complex keyword pattern detected in prompt. Classifying as 'complex'.")
            return True

        # Rule 4: Otherwise, classify as simple (suited for local execution)
        logger.info(f"Routing: Prompt is short ({prompt_len} chars) with no complexity markers. Classifying as 'simple'.")
        return False
