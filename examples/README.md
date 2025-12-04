# Examples Directory

This directory contains practical examples demonstrating various GenAI concepts and techniques.

## Structure

### Basic Examples (`basic/`)
Start here if you're new to GenAI:

- **`simple_api_call.py`**: Basic OpenAI API usage
- **`prompt_engineering.py`**: Learn effective prompting techniques

### Intermediate Examples (`intermediate/`)
More complex patterns and integrations:

- **`langchain_basics.py`**: Introduction to LangChain framework
- **`conversation_memory.py`**: Building conversational AI with memory

### Advanced Examples (`advanced/`)
Production-ready patterns and architectures:

- **`rag_example.py`**: Retrieval-Augmented Generation pattern

## Running the Examples

1. Make sure you've completed the setup (see main README.md)

2. Activate your virtual environment:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3. Run any example:
```bash
python examples/basic/simple_api_call.py
python examples/intermediate/langchain_basics.py
python examples/advanced/rag_example.py
```

## Prerequisites

- Python 3.8+
- All dependencies installed (`pip install -r requirements.txt`)
- OpenAI API key configured in `.env` file

## Example Output

Each example includes:
- Clear console output showing what's happening
- Explanations of key concepts
- Error handling with helpful messages

## Customization

Feel free to modify these examples:
- Change the prompts to test different use cases
- Adjust temperature and max_tokens parameters
- Add your own business logic
- Experiment with different models

## Learning Path

Recommended order:
1. `basic/simple_api_call.py` - Understand the basics
2. `basic/prompt_engineering.py` - Master prompting
3. `intermediate/langchain_basics.py` - Learn frameworks
4. `intermediate/conversation_memory.py` - Add state
5. `advanced/rag_example.py` - Build with knowledge bases

## Need Help?

- Check the documentation in `docs/`
- Review error messages carefully
- Ensure your API key is set correctly
- Verify all dependencies are installed

## Contributing

To add new examples:
1. Follow the existing code structure
2. Include clear comments and docstrings
3. Add error handling
4. Update this README

Happy learning! 🚀
