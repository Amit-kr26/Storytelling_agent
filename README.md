# AI Novel Generator

This project leverages AI to generate novels based on randomly generated or user-specified topics. It uses the Gemini/OPENAI/Claude API for content generation and includes both a command-line interface and a web interface built with FastAPI. The project is automated to run daily via GitHub Actions, producing a new novel each day.

## Features

- Web Interface: Generate novels through a user-friendly web app.
- Command-Line Interface: Generate novels via a script, ideal for automation.
- Daily Automation: GitHub Actions generate a new novel daily at 12:00 AM IST.
- Logging: Logs are saved for debugging and monitoring.


### Folder Structure

- ``.github/workflows/daily_novel.yml``: Configuration for the GitHub Actions workflow that automates daily novel generation.
- ``storytelling_agent/``: Core modules for story generation.
  - ``plans.py``: Manages the story structure and planning.
  - ``prompts.py``: Defines prompts used for AI interactions.
  - ``storytelling_agent.py``: Contains the main StoryAgent class responsible for generating the story.
  - ``utils.py``: Utility functions, including text processing.

- ``app.py``: FastAPI web application for generating novels via a web interface.
- ``main.py``: Command-line script for generating novels, used by GitHub Actions.
- ``novels/``: Directory where generated novels are saved.
- ``static/``: Static files (e.g., CSS) for the web interface.
- ``templates/``: HTML templates for the web interface.
