from app.router import PromptRouter

def test_router_simple_prompts():
    router = PromptRouter(complexity_threshold_chars=50)
    # Simple, short prompts with no complex keywords
    assert router.is_complex("Hello world") is False
    assert router.is_complex("What is your name?") is False

def test_router_complex_prompts_by_length():
    router = PromptRouter(complexity_threshold_chars=20)
    # Long prompts should be classified as complex
    assert router.is_complex("This is a very long prompt that will exceed the twenty character limit.") is True

def test_router_complex_prompts_by_keywords():
    router = PromptRouter(complexity_threshold_chars=100)
    # Prompts containing reasoning keywords should be complex
    assert router.is_complex("Can you solve this math problem?") is True
    assert router.is_complex("architect a scalable API") is True
    assert router.is_complex("debug this python crash error") is True
