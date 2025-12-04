# Setup Instructions

This guide provides detailed instructions for setting up your development environment for GenAI projects.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**: [Download Python](https://www.python.org/downloads/)
- **pip**: Python package installer (usually comes with Python)
- **Git**: Version control system [Download Git](https://git-scm.com/downloads)
- **Text Editor or IDE**: VS Code, PyCharm, or your preferred editor

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ajaykumarbk75/genAI_philips_Nov_Dec2025.git
cd genAI_philips_Nov_Dec2025
```

### 2. Create a Virtual Environment

Creating a virtual environment isolates your project dependencies.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when the virtual environment is activated.

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- OpenAI Python library
- LangChain and related packages
- Jupyter for notebooks
- Supporting libraries (numpy, pandas, etc.)

### 4. Set Up API Keys

#### Get Your OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (you won't be able to see it again!)

#### Configure Environment Variables

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important**: Never commit your `.env` file to version control. It's already in `.gitignore`.

### 5. Verify Installation

Test that everything is set up correctly:

```bash
python -c "import openai; print('OpenAI library installed successfully')"
python -c "import langchain; print('LangChain installed successfully')"
```

### 6. Run Your First Example

```bash
python examples/basic/simple_api_call.py
```

If everything is set up correctly, you should see a response from the OpenAI API.

## IDE Setup

### Visual Studio Code

Recommended extensions:
- Python (by Microsoft)
- Pylance
- Jupyter
- Python Docstring Generator

### PyCharm

1. Open the project folder
2. Configure the virtual environment as the project interpreter
3. Enable Jupyter support (Professional edition)

## Jupyter Notebook Setup

To use the Jupyter notebooks:

```bash
jupyter notebook
```

This will open Jupyter in your browser. Navigate to the `notebooks/` directory.

## Troubleshooting

### "Module not found" errors

Make sure your virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### OpenAI API errors

- Verify your API key is correct in `.env`
- Check you have credits in your OpenAI account
- Ensure the `.env` file is in the project root directory

### Permission errors on macOS/Linux

You might need to use `python3` instead of `python`:
```bash
python3 -m venv venv
```

### SSL Certificate errors

On some corporate networks, you might need to:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Optional: Additional Tools

### For Advanced Users

1. **Poetry** (alternative to pip):
```bash
pip install poetry
poetry install
```

2. **Pre-commit hooks** (for code quality):
```bash
pip install pre-commit
pre-commit install
```

3. **Testing tools**:
```bash
pip install pytest pytest-cov
```

## Next Steps

Now that your environment is set up:

1. Review [Getting Started Guide](getting-started.md)
2. Try the examples in `examples/basic/`
3. Explore Jupyter notebooks
4. Read [Best Practices](best-practices.md)

## Keeping Your Environment Updated

Periodically update your dependencies:

```bash
pip install --upgrade -r requirements.txt
```

## Deactivating the Virtual Environment

When you're done working:

```bash
deactivate
```

## Additional Resources

- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)
- [OpenAI Python Library Documentation](https://github.com/openai/openai-python)
- [LangChain Installation Guide](https://python.langchain.com/docs/get_started/installation)

If you encounter any issues not covered here, please check the main README or consult the official documentation for the specific tool or library.
