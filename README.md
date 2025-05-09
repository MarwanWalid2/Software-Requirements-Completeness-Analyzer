# Requirements Analyzer with Multi-LLM Support

A software requirements analysis tool that uses multiple LLMs (Large Language Models) to generate domain models, detect missing requirements, and analyze requirement completeness.

## Features

- **Multi-LLM Support**: Choose from multiple LLMs (DeepSeek, OpenAI, Claude) for analysis
- **Majority Voting**: Aggregate results from multiple LLMs using different voting strategies
- **Domain Model Generation**: Generate UML class diagrams from natural language requirements
- **Requirements Analysis**: Detect issues, missing requirements, and measure completeness
- **Interactive UI**: Review and accept/reject suggested improvements

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/requirements-analyzer.git
   cd requirements-analyzer
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Edit `.env` and add your API keys for the LLMs you want to use:
     * DeepSeek: Get API key from https://platform.deepseek.com
     * OpenAI: Get API key from https://platform.openai.com
     * Claude: Get API key from https://console.anthropic.com

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Input your software requirements in the text area.

4. Select the LLM models you want to use for analysis.

5. Choose the aggregation method for combining results from multiple models.

6. Click "Analyze Requirements" to generate the domain model and analysis.

7. Review the generated model and suggestions, then accept or reject changes.

## Multi-LLM Configuration

The system supports three LLM providers:

1. **DeepSeek**: Uses the DeepSeek chat model
2. **OpenAI**: Uses the GPT models (default: gpt-o4-mini (high reasoning))
3. **Claude**: Uses Claude models (default: claude-3-7-sonnet-20250219)

### Aggregation Methods

When multiple LLMs are selected, their results can be combined using different methods:

1. **Majority Vote**: Simple majority voting where elements that appear in most models are included
2. **Weighted Vote**: Similar to majority voting but with model-specific weights
3. **LLM-based Meta-Analysis**: Use another LLM to analyze and combine the results from other models

## Folder Structure

- `app.py`: Main application entry point
- `config.py`: Configuration and API key management
- `models/`: Domain model analysis logic
- `services/`: LLM clients and adapters
- `templates/`: HTML templates for the UI
- `utils/`: Utility functions for JSON handling and other tasks
- `log/`: Log files

## Extending the System

### Adding a New LLM

1. Add the new API key to `.env` and `config.py`
2. Create a new client module in `services/`
3. Add a new adapter class in `services/llm_adapters.py`
4. Update the factory in `services/llm_client_factory.py`
5. Add the new model to the available models in `config.py`

## License
