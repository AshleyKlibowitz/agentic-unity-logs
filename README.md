# 🕵️‍♂️ Agentic AI Log Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
![AI](https://img.shields.io/badge/AI-Ollama%20%7C%20OpenAI-green)
![License](https://img.shields.io/badge/License-MIT-purple)

## 📖 Project Overview

This is an advanced, **Agentic AI-powered diagnostic tool** designed for game developers. Unlike traditional log parsers, this application utilizes an autonomous agent architecture to not only read logs but to **reason, plan, and execute resolution strategies**.

The system ingests raw game engine logs (Unity/Unreal/Custom), detects critical patterns (crashes, memory leaks, security risks), and uses a multi-step planning engine to autonomously generate C# code fixes and performance optimizations.

## 🌟 Key Agentic Features

This project demonstrates a sophisticated **Agentic AI Architecture**:

* **🧠 Persistent Agent Memory:** The system maintains a JSON-based memory of past interactions, allowing it to "learn" successful resolution patterns over time.
* **🎯 Autonomous Planning:** The agent analyzes log density and creates a multi-step resolution plan (e.g., *1. Fix Criticals -> 2. Address Warnings -> 3. Optimize*).
* **🔧 Tool Usage & Simulation:** The agent simulates "tool use" to validate code syntax, test fixes, and estimate performance impacts before presenting them to the user.
* **🔄 Self-Correction:** The system tracks failed attempts vs. successful feedback to refine its persona and recommendations.

## 🚀 Functional Features

### 📊 Interactive Dashboard
* **Glassmorphism UI:** A modern, visually stunning interface built with custom CSS.
* **Risk Tagging:** Automatic classification of logs into **High** (Security/Crash), **Medium** (Performance), and **Low** (Deprecation) priorities.
* **Metrics:** Real-time counters for error rates and system health.

### 🛠️ Automated Remediation
* **Code Generation:** Generates downloadable `.cs` files with specific fixes for issues like `NullReferenceException`, `Buffer Overflow`, and `Memory Leaks`.
* **Contextual Chat:** A chat interface where you can ask the agent specific questions about the log state.
* **Export:** Download resolved code files individually or as a ZIP package.

## 🏗️ System Architecture

The application is structured into three distinct layers:

1.  **The Agent Core:**
    * `AgentMemory`: Manages session history and learned patterns.
    * `AgentPlanner`: Generates the `AgentPlan` (goals, steps, dependencies).
    * `AgentTools`: Provides logic for validation and analysis.
2.  **The Intelligence Layer:**
    * **Local:** Integrates with [Ollama](https://ollama.ai/) (Llama 3.2 recommended) for privacy.
    * **Cloud:** Fallback integration for OpenAI (GPT-4o) via API key.
3.  **The UI Layer:** Streamlit-based frontend with reactive state management.

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.8+**
* [Ollama](https://ollama.ai/) (Recommended for local use) **OR** an OpenAI API Key.

### 1. Clone the Repository

git clone https://github.com/AshleyKlibowitz/agentic-log-analyzer.git cd agentic-log-analyzer

### 2. Install Dependencies

pip install streamlit pandas requests openai

### 3. AI Configuration

**Option A: Local AI (Free & Private) - Recommended**
* Install Ollama.
* Pull the model used in the code:
    ```
    ollama pull llama3.2:3b
    ```
* Ensure Ollama is running (`ollama serve`). The app detects it automatically.

**Option B: OpenAI (Cloud)**
* Create a `.streamlit/secrets.toml` file in the project root.
* Add your API key:
    ```toml
    OPENAI_API_KEY = "sk-your-api-key-here"
    ```

---

## 💻 Usage

**Run the App:**

streamlit run app.py

* **Upload Logs:** Drag and drop a `.log` or `.txt` file.
* **Autonomous Mode:** Click the "🚀 Autonomous Mode" tab to watch the agent execute its resolution plan step-by-step.
* **Download Fixes:** After the agent completes its analysis, download the generated C# fix files.

---

## 🛡️ Supported Issues

The system includes specific logic to detect and fix:

* NullReferenceException (Unity/C#)
* Buffer Overflows & Network Security Risks
* Memory Leaks / Out of Memory Exceptions
* JSON Deserialization Errors
* Array/List Index Out of Range
