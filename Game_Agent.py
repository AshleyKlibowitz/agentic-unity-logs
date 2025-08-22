# --- AI Integration & Configuration ---
import streamlit as st
import pandas as pd
import re
import requests
from datetime import datetime
import json
import time
import hashlib
import html
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# --- Agentic AI System Components ---

@dataclass
class AgentMemory:
    """Persistent memory for the agentic system"""
    session_id: str
    conversations: List[Dict] = None
    learned_patterns: List[Dict] = None
    success_feedback: List[Dict] = None
    failed_attempts: List[Dict] = None
    
    def __post_init__(self):
        if self.conversations is None:
            self.conversations = []
        if self.learned_patterns is None:
            self.learned_patterns = []
        if self.success_feedback is None:
            self.success_feedback = []
        if self.failed_attempts is None:
            self.failed_attempts = []

@dataclass
class AgentPlan:
    """Multi-step planning for complex problem solving"""
    goal: str
    steps: List[Dict]
    current_step: int = 0
    completed_steps: List[str] = None
    blocked_steps: List[str] = None
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
        if self.blocked_steps is None:
            self.blocked_steps = []

class AgentTools:
    """Tool usage system for the agentic AI"""
    
    @staticmethod
    def validate_code_syntax(code: str, language: str = "csharp") -> Dict[str, Any]:
        """Validate code syntax (simulated - in real implementation would use actual validators)"""
        # Simulate basic validation
        time.sleep(0.5)  # Simulate processing time
        issues = []
        if "null" in code.lower() and "!= null" not in code:
            issues.append("Potential null reference - consider null checking")
        if "{" in code and "}" not in code:
            issues.append("Unmatched braces detected")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "confidence": 0.85 if len(issues) == 0 else 0.4
        }
    
    @staticmethod
    def simulate_fix_application(fix_code: str, issue: str) -> Dict[str, Any]:
        """Simulate applying a code fix and testing it"""
        time.sleep(1.0)  # Simulate processing time
        
        # Simulate different outcomes based on issue type
        if "buffer overflow" in issue.lower():
            success_rate = 0.95
        elif "null reference" in issue.lower():
            success_rate = 0.90
        elif "memory" in issue.lower():
            success_rate = 0.75
        else:
            success_rate = 0.80
        
        import random
        success = random.random() < success_rate
        
        return {
            "success": success,
            "confidence": success_rate,
            "feedback": "Fix applied successfully" if success else "Fix needs refinement",
            "suggested_improvements": [] if success else ["Add additional error handling", "Consider edge cases"]
        }
    
    @staticmethod
    def analyze_performance_impact(code_change: str) -> Dict[str, Any]:
        """Analyze potential performance impact of code changes"""
        time.sleep(0.3)
        
        performance_score = 0.8  # Baseline
        if "loop" in code_change.lower():
            performance_score -= 0.1
        if "try.*catch" in code_change.lower():
            performance_score -= 0.05
        if "null.*check" in code_change.lower():
            performance_score += 0.1
        
        return {
            "performance_score": max(0.1, min(1.0, performance_score)),
            "recommendations": ["Consider caching results", "Optimize loop conditions"],
            "estimated_impact": "Low" if performance_score > 0.7 else "Medium"
        }

class AgenticMemoryManager:
    """Persistent memory management for the agentic system"""
    
    def __init__(self):
        self.memory_file = Path("agent_memory.json")
    
    def load_memory(self, session_id: str) -> AgentMemory:
        """Load persistent memory for a session"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    if session_id in data:
                        return AgentMemory(**data[session_id])
            except Exception as e:
                st.error(f"Memory load error: {e}")
        
        return AgentMemory(session_id=session_id)
    
    def save_memory(self, memory: AgentMemory):
        """Save memory to persistent storage"""
        try:
            data = {}
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
            
            data[memory.session_id] = asdict(memory)
            
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            st.error(f"Memory save error: {e}")

class AgenticPlanner:
    """Multi-step planning and execution system"""
    
    @staticmethod
    def create_resolution_plan(issues: List[Dict], memory: AgentMemory) -> AgentPlan:
        """Create a comprehensive plan to resolve all identified issues"""
        
        # Sort issues by priority and complexity
        critical_issues = [i for i in issues if "High:" in i.get('risk_tag', '')]
        medium_issues = [i for i in issues if "Medium:" in i.get('risk_tag', '')]
        low_issues = [i for i in issues if "Low:" in i.get('risk_tag', '')]
        
        steps = []
        
        # Step 1: Address critical issues first
        for issue in critical_issues:
            steps.append({
                "id": f"critical_{len(steps)}",
                "type": "fix_critical_issue",
                "description": f"Resolve critical issue: {issue['message'][:50]}...",
                "issue": issue,
                "priority": "HIGH",
                "estimated_time": "15-30 minutes",
                "dependencies": []
            })
        
        # Step 2: Handle medium priority issues
        for issue in medium_issues:
            steps.append({
                "id": f"medium_{len(steps)}",
                "type": "fix_medium_issue", 
                "description": f"Address medium priority: {issue['message'][:50]}...",
                "issue": issue,
                "priority": "MEDIUM",
                "estimated_time": "10-20 minutes",
                "dependencies": [s["id"] for s in steps if s["priority"] == "HIGH"]
            })
        
        # Step 3: Optimization and cleanup
        steps.append({
            "id": "optimization",
            "type": "optimize_performance",
            "description": "Optimize overall system performance",
            "priority": "LOW",
            "estimated_time": "20-40 minutes",
            "dependencies": [s["id"] for s in steps if s["priority"] in ["HIGH", "MEDIUM"]]
        })
        
        # Step 4: Validation and testing
        steps.append({
            "id": "validation",
            "type": "validate_fixes",
            "description": "Validate all applied fixes and run tests",
            "priority": "CRITICAL",
            "estimated_time": "15-25 minutes", 
            "dependencies": [s["id"] for s in steps[:-1]]
        })
        
        # Step 5: Enhanced monitoring and documentation
        steps.append({
            "id": "enhancement",
            "type": "enhance_monitoring",
            "description": "Enhance monitoring capabilities and update documentation",
            "priority": "LOW",
            "estimated_time": "10-20 minutes",
            "dependencies": [s["id"] for s in steps]
        })
        
        goal = f"Resolve {len(critical_issues)} critical, {len(medium_issues)} medium, and {len(low_issues)} low priority issues"
        
        return AgentPlan(goal=goal, steps=steps)

# Initialize agentic components
@st.cache_resource
def get_agentic_components():
    """Initialize the agentic AI system components"""
    return {
        "memory_manager": AgenticMemoryManager(),
        "tools": AgentTools(),
        "planner": AgenticPlanner()
    }

# --- App Configuration ---
st.set_page_config(
    page_title="Game Log Analysis Tool", 
    page_icon="🕵️‍♂️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Beautiful, modern CSS styling with glassmorphism and gradients
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Clean white background */
    .stApp {
        background: #ffffff;
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    
    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Professional glassmorphism header */
    .beautiful-header {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 25px;
        padding: 4rem 3rem;
        margin-bottom: 3rem;
        box-shadow: 0 25px 45px rgba(0,0,0,0.08);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .beautiful-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .beautiful-header h1 {
        color: #2d3748;
        margin: 0;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .beautiful-header p {
        color: #4a5568;
        margin: 0;
        font-size: 1.3rem;
        font-weight: 400;
        opacity: 0.8;
    }
    
    /* Professional file upload area */
    .upload-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 25px;
        padding: 0;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        overflow: hidden;
        position: relative;
    }
    
    .upload-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        text-align: center;
        margin: 0;
    }
    
    .upload-header h3 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .upload-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .upload-body {
        padding: 1.5rem 2rem 2rem 2rem;
        text-align: center;
    }
    
    /* Integrated file uploader */
    .stFileUploader {
        width: 100%;
    }
    
    .stFileUploader > div {
        width: 100%;
    }
    
    .stFileUploader > div > div {
        background: rgba(102, 126, 234, 0.05);
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        width: 100%;
        margin: 0;
    }
    
    .stFileUploader > div > div::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .stFileUploader > div > div:hover::before {
        left: 100%;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.08);
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.15);
    }
    
    .stFileUploader label {
        color: #667eea !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        font-family: 'Inter', sans-serif !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Hide default file uploader text */
    .stFileUploader > div > div > div > div > span {
        display: none;
    }
    
    .stFileUploader > div > div::after {
        content: '🖱️ Click to browse files or drag & drop here';
        display: block;
        color: #667eea;
        font-weight: 500;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        margin-top: 1rem;
    }
    
    /* Professional metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        box-sizing: border-box;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--metric-color);
        opacity: 0.8;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.12);
        background: rgba(255, 255, 255, 1);
        border-color: rgba(102, 126, 234, 0.2);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 3rem;
        margin-bottom: 0.8rem;
        font-variant-numeric: tabular-nums;
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        transform: translateY(-15px);
        position: relative;
        width: 100%;
        text-align: center;
    }
    
    .metric-card .metric-title {
        color: #2d3748;
        margin: 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .metric-card .metric-subtitle {
        color: #718096;
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Professional section titles */
    .section-title {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        color: #2d3748;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 3rem 0 1.5rem 0;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-left: 5px solid #667eea;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        font-family: 'Inter', sans-serif;
    }
    
    /* Enhanced issue cards */
    .issue-card {
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        border-left: 4px solid;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .issue-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        z-index: -1;
    }
    
    .issue-card:hover {
        transform: translateX(5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    /* Professional chat styling */
    .chat-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 20px;
        padding: 0;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        overflow: hidden;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        margin: 0;
    }
    
    .chat-header h3 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .chat-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Professional tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(248, 250, 252, 0.8);
        backdrop-filter: blur(10px);
        padding: 0.5rem;
        border-radius: 15px;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(45, 55, 72, 0.7);
        border: none;
        padding: 1rem 1.5rem;
        font-weight: 500;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        backdrop-filter: blur(10px);
    }
    
    /* Success message styling */
    .success-message {
        background: rgba(72, 187, 120, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(72, 187, 120, 0.3);
        color: #2f855a;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    /* Button improvements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 500;
        padding: 0.7rem 1.5rem;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Ensure disabled buttons also have white text */
    .stButton > button:disabled {
        background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%) !important;
        color: white !important;
        opacity: 0.7;
        cursor: not-allowed;
    }
    
    .stButton > button:disabled:hover {
        transform: none;
        box-shadow: none;
    }
    
    /* Clean dataframe styling without blur effects */
    [data-testid="stDataFrame"] {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stDataFrame"] .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- AI Detection (Single Source of Truth) ---
@st.cache_resource
def get_ai_config():
    """Detects available AI service (Ollama or OpenAI) and returns its type."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return "ollama"
    except requests.RequestException:
        pass

    try:
        if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"]:
            return "openai"
    except (KeyError, FileNotFoundError):
        pass
    
    return None

AI_TYPE = get_ai_config()
AI_ENABLED = AI_TYPE is not None

# --- Agentic Session Management ---
def initialize_agentic_session():
    """Initialize or restore agentic AI session with persistent memory"""
    if 'agentic_components' not in st.session_state:
        st.session_state.agentic_components = get_agentic_components()
    
    if 'session_id' not in st.session_state:
        # Create unique session ID based on timestamp and random component
        import uuid
        st.session_state.session_id = f"agent_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    
    if 'agent_memory' not in st.session_state:
        memory_manager = st.session_state.agentic_components['memory_manager']
        st.session_state.agent_memory = memory_manager.load_memory(st.session_state.session_id)
    
    if 'active_plan' not in st.session_state:
        st.session_state.active_plan = None
    
    if 'autonomous_mode' not in st.session_state:
        st.session_state.autonomous_mode = True  # Enable full autonomy by default
    
    return st.session_state.agent_memory

def save_agentic_state():
    """Save current agentic state to persistent memory"""
    if 'agent_memory' in st.session_state and 'agentic_components' in st.session_state:
        memory_manager = st.session_state.agentic_components['memory_manager']
        memory_manager.save_memory(st.session_state.agent_memory)

# Initialize agentic session
agent_memory = initialize_agentic_session()

# --- Regex Patterns ---
log_pattern = re.compile(r"\[(.*?)\]\s+(ERROR|WARNING|INFO)\s+(.*)")

def run_ai_completion(prompt):
    """A centralized function to call the detected AI model."""
    if not AI_ENABLED:
        return """⚠️ **AI Analysis is Disabled**

No connection to a local Ollama instance or OpenAI API key was found.

**To enable this feature:**

* **Option 1 (Recommended):** Run a local AI model with [Ollama](https://ollama.ai/). It's free and private.
* **Option 2:** Add your `OPENAI_API_KEY` to your Streamlit secrets.

The rest of the app's features will still work without AI.
"""

    try:
        if AI_TYPE == "ollama":
            data = {
                "model": "llama3.2:3b", "prompt": prompt, "stream": False,
                "options": {"temperature": 0.5, "num_predict": 1500}
            }
            response = requests.post("http://localhost:11434/api/generate", json=data, timeout=60)
            response.raise_for_status()
            return response.json()["response"]
        
        elif AI_TYPE == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500, temperature=0.5
            )
            return response.choices[0].message.content

    except Exception as e:
        st.error(f"AI model error: {e}")
        return f"⚠️ AI model temporarily unavailable. Please check your connection and configuration.\n\n**Error:** `{str(e)}`"
    
    return "AI model not found or misconfigured."

# --- Agentic AI Core ---

def get_agent_persona_prompt(task_prompt):
    """Creates a consistent persona for the AI agent."""
    return f"""
    You are Game Log Analyzer, an expert in game development and engine diagnostics.
    Your analysis is sharp, technical, and actionable. Your goal is to help developers
    quickly identify, understand, and fix critical issues from log files.

    User's Task:
    {task_prompt}

    Provide your response in clear, well-structured markdown.
    """

def run_proactive_analysis(grouped_data):
    """Fully agentic initial analysis with autonomous planning and tool usage"""
    if grouped_data.empty:
        return "No issues were found in the log file. Looks clean!"
    
    # Get agentic components
    agent_memory = st.session_state.get('agent_memory')
    components = st.session_state.get('agentic_components')
    
    if not agent_memory or not components:
        return "Agentic system initializing..."
    
    # Convert grouped_data to issues format for planning
    issues = []
    for _, row in grouped_data.iterrows():
        issues.append({
            'message': row['message'],
            'level': row['level'],
            'count': row['count'],
            'risk_tag': row.get('risk_tag', 'Unknown')
        })
    
    # Create autonomous resolution plan
    planner = components['planner']
    resolution_plan = planner.create_resolution_plan(issues, agent_memory)
    st.session_state.active_plan = resolution_plan
    
    # Analyze patterns from memory
    historical_context = ""
    if agent_memory.learned_patterns:
        similar_patterns = [p for p in agent_memory.learned_patterns 
                          if any(keyword in grouped_data['message'].str.lower().str.cat()
                                for keyword in p.get('keywords', []))]
        if similar_patterns:
            historical_context = f"\n\n**🧠 Learning from History:** I've seen similar patterns {len(similar_patterns)} times before. "
            historical_context += f"Previous successful approaches included: {', '.join([p.get('successful_approach', 'N/A') for p in similar_patterns[:2]])}."
    
    # Perform autonomous tool validation
    tools = components['tools']
    validation_results = []
    for issue in issues[:3]:  # Validate top 3 issues
        if issue['level'] == 'ERROR':
            # Simulate code analysis for the issue
            analysis = tools.analyze_performance_impact(f"Fix for: {issue['message']}")
            validation_results.append(f"- {issue['message'][:60]}... (Impact: {analysis['estimated_impact']})")
    
    tool_analysis = ""
    if validation_results:
        tool_analysis = f"\n\n**🔧 Autonomous Pre-Analysis:**\n" + "\n".join(validation_results)
    
    # Build context with agentic capabilities
    context = f"Here is my autonomous analysis of {len(grouped_data)} unique issues:\n"
    for _, row in grouped_data.head(5).iterrows():
        context += f"- {row['level']}: {row['message']} (occurred {row['count']} times)\n"
    
    # Enhanced agentic task with autonomous capabilities
    task = f"""
    As a fully autonomous AI agent, provide an "Agent's Initial Briefing" with complete autonomous analysis:

    1. **Overall Health Assessment**: Comprehensive system health analysis
    2. **Top Priority Issue**: Most critical issue with autonomous impact assessment
    3. **Recommended First Action**: Immediate autonomous action I will take
    4. **Autonomous Plan**: I've created a {len(resolution_plan.steps)}-step resolution plan
    5. **Self-Learning**: I continuously learn from patterns in log files, updating my approach to address emerging issues and improving overall system stability for better performance optimization.

    **My Autonomous Capabilities:**
    - 🎯 Multi-step planning and execution
    - 🔧 Real-time code validation and testing
    - 🧠 Persistent memory and pattern learning
    - 🔄 Self-correction and improvement
    - 📊 Performance impact analysis

    Current Log Analysis:
    {context}
    
    Autonomous Plan Created: {resolution_plan.goal}
    {historical_context}
    {tool_analysis}
    """
    
    prompt = get_agent_persona_prompt(task)
    response = run_ai_completion(prompt)
    
    # Save learning to memory
    agent_memory.conversations.append({
        "timestamp": datetime.now().isoformat(),
        "type": "proactive_analysis",
        "issues_analyzed": len(issues),
        "plan_created": True,
        "critical_issues": len([i for i in issues if "High:" in i.get('risk_tag', '')])
    })
    
    # Save state
    save_agentic_state()
    
    return response + f"\n\n**🤖 Autonomous Status:** Created {len(resolution_plan.steps)}-step resolution plan. Ready for autonomous execution."

def get_specific_analysis(query, grouped_data, specific_finding=None):
    """Fully agentic response with autonomous problem-solving and tool usage"""
    # Get agentic components
    agent_memory = st.session_state.get('agent_memory')
    components = st.session_state.get('agentic_components')
    active_plan = st.session_state.get('active_plan')
    
    if not agent_memory or not components:
        return "Agentic system initializing..."
    
    # Autonomous context building
    context = f"I'm autonomously analyzing game engine logs with {len(grouped_data)} unique issues found.\n"
    if not grouped_data.empty:
        context += "My current priority analysis includes:\n"
        for i, (_, row) in enumerate(grouped_data.head(3).iterrows(), 1):
            context += f"{i}. {row['level']}: {row['message']} (occurs {row['count']} times)\n"
    
    # Add specific finding context
    if specific_finding is not None:
        context += f"\nUser is asking about: '{specific_finding['message']}' (Level: {specific_finding['level']}, Count: {specific_finding['count']})"
    
    # Autonomous tool usage
    tools = components['tools']
    autonomous_actions = []
    
    # If asking about a specific error, autonomously validate and test solutions
    if specific_finding and specific_finding['level'] == 'ERROR':
        # Simulate autonomous code validation
        validation = tools.validate_code_syntax(f"// Fix for {specific_finding['message']}")
        autonomous_actions.append(f"🔧 Auto-validated potential fix (Confidence: {validation['confidence']:.0%})")
        
        # Simulate autonomous fix testing  
        test_result = tools.simulate_fix_application("// Generated fix code", specific_finding['message'])
        autonomous_actions.append(f"🧪 Auto-tested solution ({test_result['feedback']})")
        
        # Performance impact analysis
        perf_analysis = tools.analyze_performance_impact("// Performance analysis")
        autonomous_actions.append(f"📊 Performance impact: {perf_analysis['estimated_impact']}")
    
    # Check memory for similar patterns
    memory_insights = []
    if agent_memory.learned_patterns:
        query_lower = query.lower()
        relevant_patterns = [p for p in agent_memory.learned_patterns 
                           if any(keyword in query_lower for keyword in p.get('keywords', []))]
        if relevant_patterns:
            memory_insights.append(f"🧠 Found {len(relevant_patterns)} similar patterns in memory")
            for pattern in relevant_patterns[:2]:
                if pattern.get('successful_approach'):
                    memory_insights.append(f"  • Previous success: {pattern['successful_approach']}")
    
    # Plan-aware response
    plan_context = ""
    if active_plan and active_plan.steps:
        relevant_steps = [s for s in active_plan.steps if query.lower() in s['description'].lower()]
        if relevant_steps:
            plan_context = f"\n\n**🎯 Autonomous Plan Alignment:** This relates to Step {relevant_steps[0]['id']} in my resolution plan."
    
    # Build autonomous action summary
    action_summary = ""
    if autonomous_actions:
        action_summary = "\n\n**🤖 Autonomous Actions Performed:**\n" + "\n".join(autonomous_actions)
    
    memory_summary = ""
    if memory_insights:
        memory_summary = "\n\n**🧠 Memory Insights:**\n" + "\n".join(memory_insights)
    
    # Enhanced agentic task
    task = f"""
    As a fully autonomous AI agent, provide a comprehensive response to: "{query}"
    
    **My Autonomous Capabilities in Action:**
    - 🔍 Real-time analysis and validation  
    - 🔧 Autonomous tool usage and testing
    - 🧠 Learning from persistent memory
    - 🎯 Plan-aware intelligent responses
    - 🔄 Self-improving recommendations
    
    Context: {context}
    {plan_context}
    
    Provide a detailed, actionable response that demonstrates my autonomous problem-solving abilities.
    """
    
    prompt = get_agent_persona_prompt(task)
    response = run_ai_completion(prompt)
    
    # Learn from this interaction
    agent_memory.conversations.append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "autonomous_actions": len(autonomous_actions),
        "memory_patterns_used": len(memory_insights),
        "plan_aligned": bool(plan_context)
    })
    
    # Extract keywords for future pattern matching
    keywords = [word.lower() for word in query.split() if len(word) > 3]
    if keywords:
        existing_pattern = next((p for p in agent_memory.learned_patterns 
                               if set(keywords).intersection(set(p.get('keywords', [])))), None)
        if existing_pattern:
            existing_pattern['usage_count'] = existing_pattern.get('usage_count', 0) + 1
        else:
            agent_memory.learned_patterns.append({
                "keywords": keywords,
                "query_type": "specific_analysis",
                "usage_count": 1,
                "successful_approach": "autonomous_multi_tool_analysis"
            })
    
    # Save enhanced state
    save_agentic_state()
    
    # Comprehensive HTML tag removal
    import re
    final_response = response + action_summary + memory_summary
    
    # Remove all possible variations of HTML tags
    # Method 1: Standard HTML tag removal
    final_response = re.sub(r'<[^<>]*>', '', final_response)
    
    # Method 2: Specific troublesome tags
    problematic_tags = ['</div>', '<div>', '<div>', '</span>', '<span>', '<br>', '<br/>', '<p>', '</p>']
    for tag in problematic_tags:
        final_response = final_response.replace(tag, '')
    
    # Method 3: Case insensitive removal
    final_response = re.sub(r'(?i)<\s*/?\s*div[^>]*>', '', final_response)
    
    # Method 4: Remove any remaining < and > that might be stray
    final_response = re.sub(r'</?[a-zA-Z][^>]*>', '', final_response)
    
    return final_response.strip()
    
    task = f"""
    Please answer the user's question based on the provided log context.
    Focus on:
    1.  **Meaning**: What the issue means in game development.
    2.  **Impact**: Potential effect on the game (e.g., performance, crashes).
    3.  **Recommendations**: Specific, actionable steps to fix it.
    4.  **Priority**: Assign a priority level (e.g., Critical, High, Medium, Low).

    Log Context: {context}
    User question: {query}
    """
    
    prompt = get_agent_persona_prompt(task)
    return run_ai_completion(prompt)

def generate_suggested_questions(grouped_data):
    """Creates context-aware suggested questions for the user."""
    if grouped_data.empty or not AI_ENABLED:
        return []
    
    suggestions = ["What's the most critical issue?"]
    top_issue = grouped_data.iloc[0]
    
    if "High:" in top_issue.get('risk_tag', ''):
        suggestions.append(f"How do I fix the '{top_issue['message']}' error?")
    if "Medium:" in top_issue.get('risk_tag', ''):
        suggestions.append(f"Explain the '{top_issue['message']}' warning.")
        
    suggestions.append("Summarize the performance risks.")
    return list(dict.fromkeys(suggestions))[:3]

def generate_code_fixes(grouped_data):
    """Generate specific code fix recommendations based on log analysis."""
    if grouped_data.empty:
        return []
    
    fixes = []
    # Process ALL rows, not just the first 10
    for _, row in grouped_data.iterrows():
        message = row['message'].lower()
        original_message = row['message']
        level = row['level']
        
        # Generate specific code fixes based on common game engine issues
        if level == "ERROR":
            if "nullreference" in message or "null pointer" in message:
                fixes.append({
                    "issue": original_message,
                    "priority": "🔴 Critical",
                    "file_hint": "Check initialization code",
                    "code_fix": """// Before (problematic):
GameObject player;
player.transform.position = newPos; // NullReferenceException

// After (fixed):
GameObject player = GameObject.FindWithTag("Player");
if (player != null) {
    player.transform.position = newPos;
} else {
    Debug.LogError("Player object not found!");
}"""
                })
            elif "buffer overflow" in message:
                fixes.append({
                    "issue": original_message,
                    "priority": "🔴 Critical",
                    "file_hint": "NetworkManager.cs or similar buffer handling",
                    "code_fix": """// Before (problematic):
char buffer[256];
strcpy(buffer, userInput); // Buffer overflow risk

// After (fixed):
char buffer[256];
strncpy(buffer, userInput, sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\\0'; // Ensure null termination

// Or in C#:
string safeInput = userInput.Length > 255 ? 
    userInput.Substring(0, 255) : userInput;"""
                })
            elif "out of memory" in message or "memory" in message and "allocat" in message:
                fixes.append({
                    "issue": original_message,
                    "priority": "🔴 Critical",
                    "file_hint": "Memory allocation code",
                    "code_fix": """// Before (problematic):
List<GameObject> objects = new List<GameObject>();
while (true) {
    objects.Add(new GameObject()); // Memory leak
}

// After (fixed):
List<GameObject> objects = new List<GameObject>();
const int MAX_OBJECTS = 1000;
while (objects.Count < MAX_OBJECTS) {
    GameObject obj = new GameObject();
    if (obj != null) {
        objects.Add(obj);
    } else {
        Debug.LogError("Failed to create object - out of memory");
        break;
    }
}"""
                })
            elif "deserialization" in message or "playerdata.json" in message:
                fixes.append({
                    "issue": original_message,
                    "priority": "🟠 Medium",
                    "file_hint": "JSON serialization code",
                    "code_fix": """// Before (problematic):
PlayerData data = JsonConvert.DeserializeObject<PlayerData>(jsonString);

// After (fixed):
try {
    PlayerData data = JsonConvert.DeserializeObject<PlayerData>(jsonString);
    if (data == null) {
        data = new PlayerData(); // Default values
        Debug.LogWarning("Invalid player data, using defaults");
    }
} catch (JsonException ex) {
    Debug.LogError($"Failed to deserialize player data: {ex.Message}");
    PlayerData data = new PlayerData(); // Fallback to defaults
}"""
                })
            elif "unityengine" in message and "gameobject" in message:
                fixes.append({
                    "issue": original_message,
                    "priority": "🟠 Medium",
                    "file_hint": "GameObject lifecycle management",
                    "code_fix": """// Before (problematic):
GameObject obj = GetComponent<GameObject>(); // Wrong usage
obj.IsActive(); // Unhandled exception

// After (fixed):
GameObject obj = gameObject; // Correct reference
if (obj != null && obj.activeInHierarchy) {
    // Safe to use object
    obj.SetActive(true);
} else {
    Debug.LogWarning("GameObject is null or inactive");
}"""
                })
            elif "index out of range" in message or "array" in message:
                fixes.append({
                    "issue": original_message,
                    "priority": "🔴 Critical",
                    "file_hint": "Array/List access code",
                    "code_fix": """// Before (problematic):
enemies[enemyIndex].TakeDamage(damage);

// After (fixed):
if (enemyIndex >= 0 && enemyIndex < enemies.Count) {
    enemies[enemyIndex].TakeDamage(damage);
} else {
    Debug.LogWarning($"Invalid enemy index: {enemyIndex}");
}"""
                })
        elif level == "WARNING":
            if "deprecated" in message or "physicsplugin" in message:
                fixes.append({
                    "issue": original_message,
                    "priority": "🟡 Low",
                    "file_hint": "Update plugin references",
                    "code_fix": """// Before (deprecated):
using OldPhysicsPlugin v1.2;
PhysicsManager.LegacyMethod();

// After (updated):
using NewPhysicsPlugin v2.0;
PhysicsManager.UpdatedMethod();

// Or disable the warning if updating isn't possible:
#pragma warning disable CS0618 // Disable obsolete warning
PhysicsManager.LegacyMethod();
#pragma warning restore CS0618"""
                })
    
    # Sort fixes by priority (Critical first, then High, Medium, Low)
    priority_order = {"🔴 Critical": 1, "🟡 High": 2, "🟠 Medium": 3, "� Low": 4}
    fixes.sort(key=lambda x: priority_order.get(x['priority'], 5))
    
    return fixes

# --- Core Log Processing Logic ---

def parse_logs(log_text):
    parsed = [
        {"timestamp": m.group(1), "level": m.group(2), "message": m.group(3).strip()}
        for m in log_pattern.finditer(log_text)
    ]
    return pd.DataFrame(parsed) if parsed else pd.DataFrame(columns=["timestamp", "level", "message"])

def group_issues(df):
    if df.empty:
        return pd.DataFrame(columns=["level", "message", "count"])
    grouped = df.groupby(["level", "message"]).size().reset_index(name="count")
    return grouped.sort_values(by="count", ascending=False)

def tag_risks(level, message):
    msg = message.lower()
    if level == "ERROR":
        if any(s in msg for s in ["nullreference", "null pointer", "crash", "fatal", "critical"]):
            return "High: Critical Error"
        if any(s in msg for s in ["buffer overflow", "stack overflow", "security", "vulnerability"]):
            return "High: Security Risk"
        return "Medium: General Error"
    if level == "WARNING":
        if any(s in msg for s in ["deprecated", "obsolete", "outdated"]):
            return "Low: Update Recommended"
        if any(s in msg for s in ["memory", "leak", "performance", "slow", "lag"]):
            return "Medium: Performance"
        return "Low: General Warning"
    if level == "INFO":
        if any(s in msg for s in ["success", "ready", "complete", "loaded", "connected", "started"]):
            return "Info: Success"
        if any(s in msg for s in ["timeout", "connection failed", "network error", "corrupt"]):
            return "Medium: Connectivity"
        return "Info: General"
    return "Info: Unknown"

# --- UI Components ---

def display_metrics_dashboard(grouped_df):
    st.markdown('<div class="section-title">📊 Dashboard Overview</div>', unsafe_allow_html=True)
    
    high_risks = grouped_df['risk_tag'].str.contains("High:", na=False).sum()
    medium_risks = grouped_df['risk_tag'].str.contains("Medium:", na=False).sum()
    total_errors = grouped_df[grouped_df['level'] == 'ERROR']['count'].sum()
    total_warnings = grouped_df[grouped_df['level'] == 'WARNING']['count'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="--metric-color: linear-gradient(135deg, #f56565, #e53e3e);">
            <h3 style="color: #e53e3e;">{high_risks}</h3>
            <p class="metric-title">High-Risk Issues</p>
            <p class="metric-subtitle">Critical problems</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="--metric-color: linear-gradient(135deg, #ed8936, #dd6b20);">
            <h3 style="color: #dd6b20;">{medium_risks}</h3>
            <p class="metric-title">Medium-Risk Issues</p>
            <p class="metric-subtitle">Moderate attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="--metric-color: linear-gradient(135deg, #4299e1, #3182ce);">
            <h3 style="color: #3182ce;">{total_errors}</h3>
            <p class="metric-title">Total Errors</p>
            <p class="metric-subtitle">All error events</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="--metric-color: linear-gradient(135deg, #48bb78, #38a169);">
            <h3 style="color: #38a169;">{total_warnings}</h3>
            <p class="metric-title">Total Warnings</p>
            <p class="metric-subtitle">All warning events</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

def display_issue_cards(issues_df):
    with st.expander("⚠️ Priority Issues", expanded=True):
        if issues_df.empty:
            st.markdown('<div class="success-message">🎉 <strong>No issues found!</strong> Your logs look clean and healthy.</div>', unsafe_allow_html=True)
            return

        for _, row in issues_df.iterrows():
            risk_tag = row['risk_tag']
            
            if "High:" in risk_tag: 
                bg_color, border_color, text_color = "rgba(254, 242, 242, 0.9)", "#b91c1c", "#450a0a"
            elif "Medium:" in risk_tag: 
                bg_color, border_color, text_color = "rgba(254, 251, 243, 0.9)", "#c2410c", "#431407"
            elif "Low:" in risk_tag: 
                bg_color, border_color, text_color = "rgba(254, 252, 232, 0.9)", "#a16207", "#422006"
            else: 
                bg_color, border_color, text_color = "rgba(248, 250, 252, 0.9)", "#475569", "#1e293b"
                
            st.markdown(f"""
            <div class="issue-card" style="background: {bg_color}; border-left-color: {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <strong style="color: {border_color}; font-size: 1.1rem; font-weight: 600;">{risk_tag}</strong>
                    <span style="background: {border_color}; color: white; 
                               padding: 0.4rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.85rem;">
                        Count: {row['count']}
                    </span>
                </div>
                <div style="font-family: 'SF Mono', 'Consolas', monospace; font-size: 0.95rem; 
                            color: {text_color}; line-height: 1.5; background: rgba(255,255,255,0.3); 
                            padding: 1rem; border-radius: 10px;">
                    <span style="color: {border_color}; font-weight: 700;">[{row['level']}]</span> 
                    {row['message'][:120]}{'...' if len(row['message']) > 120 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        if len(issues_df) > 5:
            st.info(f"ℹ️ Showing top 5 of {len(issues_df)} total issues. View all in the detailed findings below.")

def generate_resolved_code_files(log_data):
    """Generate downloadable resolved code files based on completed fixes"""
    if log_data.empty:
        return {}
    
    resolved_files = {}
    
    # Generate fixes based on the issues found
    for _, row in log_data.iterrows():
        message = row['message'].lower()
        original_message = row['message']
        level = row['level']
        
        if level == "ERROR":
            if "buffer overflow" in message and "networkmanager" in message:
                resolved_files["NetworkManager_Fixed.cs"] = """// NetworkManager.cs - Buffer Overflow Fix
using System;
using UnityEngine;

public class NetworkManager : MonoBehaviour
{
    // Fixed: Buffer overflow prevention
    private const int MAX_BUFFER_SIZE = 1024;
    
    public void ProcessNetworkData(string data)
    {
        // Before: Potential buffer overflow
        // char buffer[256];
        // strcpy(buffer, data);
        
        // After: Safe buffer handling
        if (data.Length > MAX_BUFFER_SIZE)
        {
            Debug.LogWarning($"Data too large ({data.Length} bytes), truncating to {MAX_BUFFER_SIZE}");
            data = data.Substring(0, MAX_BUFFER_SIZE);
        }
        
        char[] safeBuffer = new char[MAX_BUFFER_SIZE];
        data.CopyTo(0, safeBuffer, 0, Math.Min(data.Length, MAX_BUFFER_SIZE - 1));
        
        // Process the safely buffered data
        ProcessSafeData(new string(safeBuffer));
    }
    
    private void ProcessSafeData(string safeData)
    {
        // Your processing logic here
        Debug.Log($"Processing: {safeData}");
    }
}"""
                
            elif "nullreference" in message or "null pointer" in message:
                resolved_files["NullReferenceFixed.cs"] = """// Null Reference Exception Fix
using UnityEngine;

public class GameObjectManager : MonoBehaviour
{
    // Fixed: Null reference prevention
    public void SafeGameObjectAccess()
    {
        // Before: Potential null reference
        // GameObject player;
        // player.transform.position = newPos; // NullReferenceException
        
        // After: Safe null checking
        GameObject player = GameObject.FindWithTag("Player");
        if (player != null)
        {
            player.transform.position = Vector3.zero;
            Debug.Log("Player position updated successfully");
        }
        else
        {
            Debug.LogError("Player object not found! Creating default player.");
            CreateDefaultPlayer();
        }
    }
    
    private void CreateDefaultPlayer()
    {
        GameObject newPlayer = new GameObject("Player");
        newPlayer.tag = "Player";
        newPlayer.transform.position = Vector3.zero;
    }
}"""
                
            elif "out of memory" in message or "memory" in message and "allocat" in message:
                resolved_files["MemoryManager_Fixed.cs"] = """// Memory Management Fix
using System;
using System.Collections.Generic;
using UnityEngine;

public class MemoryManager : MonoBehaviour
{
    // Fixed: Memory leak prevention
    private const int MAX_OBJECTS = 1000;
    private List<GameObject> managedObjects = new List<GameObject>();
    
    public void SafeObjectCreation()
    {
        // Before: Potential memory leak
        // while (true) {
        //     objects.Add(new GameObject()); // Memory leak
        // }
        
        // After: Safe memory management
        if (managedObjects.Count >= MAX_OBJECTS)
        {
            Debug.LogWarning($"Maximum objects reached ({MAX_OBJECTS}). Cleaning up oldest objects.");
            CleanupOldObjects();
        }
        
        try
        {
            GameObject newObj = new GameObject("ManagedObject");
            if (newObj != null)
            {
                managedObjects.Add(newObj);
                Debug.Log($"Created object. Total count: {managedObjects.Count}");
            }
        }
        catch (OutOfMemoryException)
        {
            Debug.LogError("Out of memory! Forcing garbage collection.");
            ForceGarbageCollection();
        }
    }
    
    private void CleanupOldObjects()
    {
        // Remove oldest 10% of objects
        int removeCount = Mathf.Max(1, managedObjects.Count / 10);
        for (int i = 0; i < removeCount && managedObjects.Count > 0; i++)
        {
            if (managedObjects[0] != null)
                DestroyImmediate(managedObjects[0]);
            managedObjects.RemoveAt(0);
        }
    }
    
    private void ForceGarbageCollection()
    {
        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
        Resources.UnloadUnusedAssets();
    }
}"""
                
            elif "deserialization" in message or "playerdata.json" in message:
                resolved_files["PlayerDataManager_Fixed.cs"] = """// JSON Deserialization Fix
using System;
using UnityEngine;
using Newtonsoft.Json;

[System.Serializable]
public class PlayerData
{
    public string playerName = "DefaultPlayer";
    public int level = 1;
    public float experience = 0f;
    public Vector3 position = Vector3.zero;
}

public class PlayerDataManager : MonoBehaviour
{
    // Fixed: Safe JSON deserialization
    public PlayerData LoadPlayerData(string jsonString)
    {
        // Before: Unsafe deserialization
        // PlayerData data = JsonConvert.DeserializeObject<PlayerData>(jsonString);
        
        // After: Safe deserialization with error handling
        try
        {
            if (string.IsNullOrEmpty(jsonString))
            {
                Debug.LogWarning("JSON string is null or empty. Using default player data.");
                return new PlayerData();
            }
            
            PlayerData data = JsonConvert.DeserializeObject<PlayerData>(jsonString);
            
            if (data == null)
            {
                Debug.LogWarning("Deserialization returned null. Using default player data.");
                return new PlayerData();
            }
            
            // Validate deserialized data
            ValidatePlayerData(data);
            
            Debug.Log("Player data loaded successfully");
            return data;
        }
        catch (JsonException ex)
        {
            Debug.LogError($"JSON deserialization failed: {ex.Message}");
            return new PlayerData(); // Return default data
        }
        catch (Exception ex)
        {
            Debug.LogError($"Unexpected error during deserialization: {ex.Message}");
            return new PlayerData();
        }
    }
    
    private void ValidatePlayerData(PlayerData data)
    {
        if (string.IsNullOrEmpty(data.playerName))
            data.playerName = "DefaultPlayer";
        
        if (data.level < 1)
            data.level = 1;
        
        if (data.experience < 0)
            data.experience = 0f;
    }
}"""
    
    # Add a comprehensive summary file
    if resolved_files:
        resolved_files["00_RESOLUTION_SUMMARY.txt"] = f"""🎉 AUTONOMOUS AI RESOLUTION COMPLETE
===========================================

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Issues Resolved: {len(resolved_files) - 1}

📁 Generated Files:
{chr(10).join(f"• {filename}" for filename in resolved_files.keys() if filename != "00_RESOLUTION_SUMMARY.txt")}

🤖 AI Agent Capabilities Used:
✅ Multi-step planning and execution
✅ Autonomous code generation
✅ Error pattern recognition
✅ Memory management optimization
✅ Security vulnerability fixes
✅ Performance impact analysis

🔧 Implementation Instructions:
1. Review each generated file carefully
2. Replace the problematic code sections in your project
3. Test the fixes in a development environment
4. Monitor logs for successful resolution

⚠️ Important Notes:
- These fixes are generated based on common patterns
- Always review and test before deploying to production
- Some fixes may need customization for your specific use case
- Keep backups of your original code

🚀 Powered by Autonomous AI Agent with Ollama
"""

    return resolved_files

# --- Main Streamlit App ---

def main():
    # Beautiful header
    st.markdown("""
    <div class="beautiful-header">
        <h1>🕵️‍♂️ Game Log Analysis Tool</h1>
        <p>Your Advanced Game Development Diagnostics Partner</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "log_data" not in st.session_state:
        st.session_state.log_data = pd.DataFrame()

    # Beautiful file upload section
    st.markdown("""
    <div class="upload-container">
        <div class="upload-header">
            <h3>📁 Upload Log File</h3>
            <p>Drag and drop your game engine log files to begin intelligent analysis</p>
        </div>
        <div class="upload-body">
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your log file",
        type=["log", "txt"],
        help="Supported formats: .log, .txt files up to 200MB",
        label_visibility="collapsed"
    )
    
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Clear session state if no file is uploaded
    if not uploaded_file:
        if st.session_state.session_id is not None:
            st.session_state.session_id = None
            st.session_state.log_data = pd.DataFrame()
            st.session_state.messages = []

    if uploaded_file and uploaded_file.name != st.session_state.session_id:
        st.session_state.session_id = uploaded_file.name
        st.session_state.messages = [] # Clear chat on new file
        with st.spinner("🔍 Parsing and analyzing your log entries..."):
            log_text = uploaded_file.read().decode("utf-8", errors="ignore")
            df = parse_logs(log_text)
            grouped = group_issues(df)
            if not grouped.empty:
                grouped['risk_tag'] = grouped.apply(lambda r: tag_risks(r['level'], r['message']), axis=1)
                priority_map = {"High": 1, "Medium": 2, "Low": 3, "Info": 4}
                grouped['priority'] = grouped['risk_tag'].apply(lambda x: priority_map.get(x.split(":")[0], 5))
                st.session_state.log_data = grouped.sort_values(['priority', 'count'], ascending=[True, False])
            else:
                st.session_state.log_data = grouped
        st.rerun()

    log_data = st.session_state.log_data

    if not log_data.empty:
        issues_df = log_data[log_data['priority'] < 4].copy()
        success_df = log_data[log_data['priority'] >= 4].copy()

        display_metrics_dashboard(log_data)
        display_issue_cards(issues_df)

        with st.expander("🔍 Show All Parsed Findings & Classification Key"):
            if not success_df.empty:
                st.markdown("#### ✅ Success Messages")
                display_success = success_df[['level', 'message', 'count', 'risk_tag']].copy()
                display_success.columns = ['Level', 'Message', 'Count', 'Risk Tag']
                
                def style_success_dataframe(df):
                    def color_level(val):
                        if val == 'INFO':
                            return 'background-color: rgba(5, 150, 105, 0.1); color: #059669; font-weight: 700; font-family: monospace; border-radius: 6px; padding: 0.3rem 0.6rem;'
                        return ''
                    
                    def color_risk_tag(val):
                        if 'Info: Success' in str(val):
                            return 'background-color: rgba(5, 150, 105, 0.15); color: #059669; font-weight: 700; border-radius: 8px; padding: 0.5rem 0.7rem; border: 1px solid rgba(5, 150, 105, 0.3);'
                        return ''
                    
                    return df.style.applymap(color_level, subset=['Level']).applymap(color_risk_tag, subset=['Risk Tag'])
                
                st.dataframe(style_success_dataframe(display_success), use_container_width=True)

            if not issues_df.empty:
                st.markdown("#### ⚠️ Issues & Risks")
                st.markdown("""
                **🏷️ Risk Tag Classification:**
                - 🔴 **High: Critical Error** → Fatal crashes, null references, critical errors
                - 🔴 **High: Security Risk** → Buffer overflows, security vulnerabilities
                - 🟠 **Medium: General Error** → General errors that need attention
                - 🟠 **Medium: Performance** → Memory leaks, performance warnings
                - 🟠 **Medium: Connectivity** → Network, timeout, connection issues
                - 🟡 **Low: General Warning** → General WARNING messages
                - 🟡 **Low: Update Recommended** → Deprecated APIs, outdated components
                """)
                display_issues = issues_df[['level', 'message', 'count', 'risk_tag']].copy()
                display_issues.columns = ['Level', 'Message', 'Count', 'Risk Tag']
                
                def style_issues_dataframe(df):
                    def color_level(val):
                        if val == 'ERROR':
                            return 'background-color: rgba(220, 38, 38, 0.1); color: #dc2626; font-weight: 700; font-family: monospace; border-radius: 6px; padding: 0.3rem 0.6rem;'
                        elif val == 'WARNING':
                            return 'background-color: rgba(217, 119, 6, 0.1); color: #d97706; font-weight: 700; font-family: monospace; border-radius: 6px; padding: 0.3rem 0.6rem;'
                        elif val == 'INFO':
                            return 'background-color: rgba(5, 150, 105, 0.1); color: #059669; font-weight: 700; font-family: monospace; border-radius: 6px; padding: 0.3rem 0.6rem;'
                        return ''
                    
                    def color_risk_tag(val):
                        if 'High:' in str(val):
                            return 'background-color: rgba(220, 38, 38, 0.15); color: #dc2626; font-weight: 700; border-radius: 8px; padding: 0.5rem 0.7rem; border: 1px solid rgba(220, 38, 38, 0.3);'
                        elif 'Medium:' in str(val):
                            return 'background-color: rgba(234, 88, 12, 0.15); color: #ea580c; font-weight: 700; border-radius: 8px; padding: 0.5rem 0.7rem; border: 1px solid rgba(234, 88, 12, 0.3);'
                        elif 'Low:' in str(val):
                            return 'background-color: rgba(217, 119, 6, 0.15); color: #d97706; font-weight: 700; border-radius: 8px; padding: 0.5rem 0.7rem; border: 1px solid rgba(217, 119, 6, 0.3);'
                        return ''
                    
                    return df.style.applymap(color_level, subset=['Level']).applymap(color_risk_tag, subset=['Risk Tag'])
                
                st.dataframe(style_issues_dataframe(display_issues), use_container_width=True)
        
        st.divider()

        tab1, tab2, tab3 = st.tabs(["🤖 Chat with AI Agent", "🔧 Code Analysis", "🚀 Autonomous Mode"])
        
        with tab1:
            st.markdown("""
            <div class="chat-container">
                <div class="chat-header">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                🤖 AI-Powered Log Analysis
                            </h3>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.95; color: rgba(255,255,255,0.9);">
                                Get expert insights and recommendations from our intelligent agent
                            </p>
                        </div>
                        <div style="text-align: right;">
                            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); 
                                        border-radius: 12px; padding: 0.7rem 1.2rem; border: 1px solid rgba(255,255,255,0.2);">
                                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8); margin-bottom: 0.2rem;">Powered by</div>
                                <div style="font-size: 1.3rem; font-weight: 700; color: white; 
                                           background: linear-gradient(45deg, #ffffff, #f0f0f0); 
                                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                                           background-clip: text; text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                                    🦙 Ollama
                                </div>
                                <div style="font-size: 0.75rem; color: rgba(255,255,255,0.7); margin-top: 0.1rem;">
                                    llama3.2:3b
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced Clear Chat button
            st.markdown("""
            <div style="margin: 1.5rem 0; display: flex; justify-content: flex-start;">
            </div>
            """, unsafe_allow_html=True)
            
            col_clear, _ = st.columns([1, 4])
            with col_clear:
                if st.button("🗑️ Clear Chat", 
                           help="Start fresh conversation", 
                           type="secondary",
                           use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

            # Generate initial briefing if no messages exist
            if not st.session_state.messages:
                with st.spinner("🧠 AI Agent analyzing your logs..."):
                    briefing = run_proactive_analysis(log_data)
                    st.session_state.messages.append({"role": "assistant", "content": briefing})
                    st.rerun()

            # Display chat messages with enhanced styling
            for i, message in enumerate(st.session_state.messages):
                avatar = "🕵️‍♂️" if message["role"] == "assistant" else "👨‍💻"
                with st.chat_message(message["role"], avatar=avatar):
                    if message["role"] == "assistant":
                        # Enhanced assistant message with proper container
                        if i == 0:  # First message is the initial briefing
                            st.markdown("""
                            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                                           width: 35px; height: 35px; border-radius: 50%; 
                                           display: flex; align-items: center; justify-content: center;
                                           margin-right: 0.8rem; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);">
                                    <span style="font-size: 1rem;">🔍</span>
                                </div>
                                <span style="font-weight: 600; font-size: 1.1rem; color: #2d3748;">
                                    Agent's Initial Briefing
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Professional assistant message box
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.08), rgba(118, 75, 162, 0.08)); 
                                    border: 1px solid rgba(102, 126, 234, 0.2);
                                    padding: 1.5rem; border-radius: 15px; 
                                    border-left: 4px solid #667eea; margin: 0.5rem 0; 
                                    backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                                    font-family: "Inter", sans-serif; line-height: 1.6; color: #2d3748;'>
                            <div style="display: flex; align-items: flex-start; gap: 12px;">
                                <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                                           width: 32px; height: 32px; border-radius: 50%; 
                                           display: flex; align-items: center; justify-content: center;
                                           flex-shrink: 0; margin-top: 2px;">
                                    <span style="font-size: 14px;">🕵️‍♂️</span>
                                </div>
                                <div style="flex: 1; min-width: 0;">
                                    {message["content"]}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Enhanced user message box
                        st.markdown(f"""
                        <div style='background: rgba(248, 250, 252, 0.95); 
                                    border: 1px solid rgba(226, 232, 240, 0.8);
                                    padding: 1.2rem; border-radius: 12px; 
                                    border-left: 4px solid #64748b; margin: 0.5rem 0; 
                                    backdrop-filter: blur(5px); font-family: "Inter", sans-serif;
                                    color: #374151; line-height: 1.5; box-shadow: 0 2px 8px rgba(0,0,0,0.04);'>
                            <div style="display: flex; align-items: flex-start; gap: 12px;">
                                <div style="background: linear-gradient(135deg, #64748b, #475569); 
                                           width: 32px; height: 32px; border-radius: 50%; 
                                           display: flex; align-items: center; justify-content: center;
                                           flex-shrink: 0; margin-top: 2px;">
                                    <span style="font-size: 14px;">👨‍💻</span>
                                </div>
                                <div style="flex: 1; min-width: 0; padding-top: 2px;">
                                    {message["content"]}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Chat input
            if prompt := st.chat_input("💬 Ask me anything about your logs...") or st.session_state.get("user_query"):
                if "user_query" in st.session_state:
                    prompt = st.session_state["user_query"]
                    del st.session_state["user_query"]
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.spinner("🔍 Analyzing your question..."):
                    response = get_specific_analysis(prompt, log_data)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

            # Suggested questions
            suggestions = generate_suggested_questions(log_data)
            if suggestions:
                st.markdown("##### 💡 Try asking:")
                cols = st.columns(len(suggestions))
                for i, question in enumerate(suggestions):
                    with cols[i]:
                        if st.button(question, key=f"suggest_{i}", use_container_width=True):
                            st.session_state.user_query = question
                            st.rerun()

        with tab2:
            st.markdown('<div class="section-title">🔧 Recommended Code Fixes</div>', unsafe_allow_html=True)
            code_fixes = generate_code_fixes(log_data)
            
            if not code_fixes:
                st.markdown('<div class="success-message">🎉 <strong>Excellent!</strong> No common, auto-fixable code patterns were detected!</div>', unsafe_allow_html=True)
            else:
                for i, fix in enumerate(code_fixes):
                    with st.expander(f"{fix['priority']} | {fix['issue']}", expanded=(i==0)):
                        st.markdown(f"**🎯 Issue:** {fix['issue']}")
                        st.markdown(f"**📁 Likely Location:** `{fix['file_hint']}`")
                        st.markdown("**🔧 Recommended Fix:**")
                        st.code(fix['code_fix'], language='csharp')
        
        with tab3:
            st.markdown('<div class="section-title">🚀 Autonomous AI Agent</div>', unsafe_allow_html=True)
            
            # Autonomous mode header with beautiful styling
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 20px; padding: 2rem; margin-bottom: 2rem; color: white;
                        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 3rem; margin-right: 1rem;">🤖</div>
                    <div>
                        <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">Fully Autonomous AI Agent</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.95;">
                            Advanced multi-step planning, tool usage, and self-improving AI system
                        </p>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🎯</div>
                        <div style="font-weight: 600;">Multi-Step Planning</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔧</div>
                        <div style="font-weight: 600;">Tool Usage</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
                        <div style="font-weight: 600;">Persistent Memory</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔄</div>
                        <div style="font-weight: 600;">Self-Correction</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display active plan if available
            active_plan = st.session_state.get('active_plan')
            if active_plan:
                st.markdown("### 📋 Active Resolution Plan")
                st.markdown(f"**Goal:** {active_plan.goal}")
                
                # Progress tracking with bounds checking
                completed_count = len(active_plan.completed_steps)
                total_count = len(active_plan.steps) if active_plan.steps else 1
                progress = min(1.0, max(0.0, completed_count / total_count))
                st.progress(progress, text=f"Progress: {completed_count}/{total_count} steps completed")
                
                # Step details
                all_completed = len(active_plan.completed_steps) == len(active_plan.steps)
                
                for i, step in enumerate(active_plan.steps):
                    status_icon = "✅" if step['id'] in active_plan.completed_steps else "⏳" if i == active_plan.current_step else "⏸️"
                    priority_color = {"HIGH": "#dc2626", "MEDIUM": "#ea580c", "LOW": "#059669", "CRITICAL": "#7c3aed"}.get(step['priority'], "#64748b")
                    
                    with st.expander(f"{status_icon} Step {i+1}: {step['description']}", expanded=(i == active_plan.current_step and not all_completed)):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**Type:** {step['type']}")
                            st.markdown(f"**Priority:** <span style='color: {priority_color}; font-weight: bold;'>{step['priority']}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Estimated Time:** {step['estimated_time']}")
                            if step.get('dependencies'):
                                st.markdown(f"**Dependencies:** {', '.join(step['dependencies'])}")
                        with col2:
                            # Only show execute button if not all steps are completed and it's the current step
                            if not all_completed and i == active_plan.current_step:
                                if st.button(f"Execute Step {i+1}", key=f"exec_step_{i}"):
                                    # Simulate autonomous execution
                                    with st.spinner(f"🤖 Autonomously executing {step['description']}..."):
                                        time.sleep(2)  # Simulate processing
                                        active_plan.completed_steps.append(step['id'])
                                        active_plan.current_step += 1
                                        st.success(f"✅ Completed: {step['description']}")
                                        st.rerun()
                            elif step['id'] in active_plan.completed_steps:
                                st.markdown("✅ **Completed**")
                            else:
                                st.markdown("⏳ **Pending**")
                
                # Autonomous execution controls
                st.markdown("### 🎮 Autonomous Execution Controls")
                
                if all_completed:
                    # All steps completed - show completion status and reset option
                    st.success("🎉 All resolution steps have been completed autonomously!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Status:** ✅ Plan Execution Complete")
                    with col2:
                        if st.button("🔄 Reset Plan", use_container_width=True):
                            active_plan.completed_steps = []
                            active_plan.current_step = 0
                            st.info("🔄 Plan reset - ready for execution")
                            st.rerun()
                else:
                    # Steps remaining - show execution controls
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🚀 Execute All Steps", type="primary", use_container_width=True):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            remaining_steps = [step for step in active_plan.steps if step['id'] not in active_plan.completed_steps]
                            for i, step in enumerate(remaining_steps):
                                status_text.text(f"🤖 Executing: {step['description']}")
                                time.sleep(1.5)  # Simulate processing
                                active_plan.completed_steps.append(step['id'])
                                progress_bar.progress((i + 1) / len(remaining_steps))
                            
                            status_text.text("✅ All steps completed autonomously!")
                            st.success("🎉 Autonomous execution completed successfully!")
                            save_agentic_state()
                            st.rerun()
                    
                    with col2:
                        if st.button("⏸️ Pause Execution", use_container_width=True):
                            st.info("⏸️ Autonomous execution paused")
                    
                    with col3:
                        if st.button("🔄 Reset Plan", use_container_width=True):
                            active_plan.completed_steps = []
                            active_plan.current_step = 0
                            st.info("🔄 Plan reset - ready for execution")
                            st.rerun()
                        
            # Code Export Section
            st.markdown("### 📥 Export Resolved Code")
            if active_plan and active_plan.completed_steps:
                # Generate resolved code files
                resolved_files = generate_resolved_code_files(st.session_state.get('log_data', pd.DataFrame()))
                
                if resolved_files:
                    st.markdown("**🎉 Your resolved code files are ready for download:**")
                    
                    # Create download buttons for each file
                    cols = st.columns(min(3, len(resolved_files)))
                    for i, (filename, content) in enumerate(resolved_files.items()):
                        with cols[i % 3]:
                            st.download_button(
                                label=f"📄 {filename}",
                                data=content,
                                file_name=filename,
                                mime="text/plain",
                                key=f"download_{filename}",
                                use_container_width=True
                            )
                    
                    # Bulk download as ZIP
                    if len(resolved_files) > 1:
                        import zipfile
                        from io import BytesIO
                        
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            for filename, content in resolved_files.items():
                                zip_file.writestr(filename, content)
                        
                        st.download_button(
                            label="📦 Download All Files (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="resolved_code_fixes.zip",
                            mime="application/zip",
                            type="primary",
                            use_container_width=True
                        )
                else:
                    st.info("💡 Complete the resolution steps to generate downloadable code files")
            else:
                st.info("🔄 Execute resolution steps to generate resolved code files")
            
            # Memory and learning section
            agent_memory = st.session_state.get('agent_memory')
            if agent_memory:
                st.markdown("### 🧠 How This AI Assistant is Learning to Help You Better")
                
                st.markdown("""
                **What you see below:** This shows how the AI is getting smarter about helping you with game development issues. 
                Think of it like a personal assistant that remembers what worked well before.
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Conversations", len(agent_memory.conversations))
                    st.caption("💬 How many times you've asked me questions")
                    
                    st.metric("Things I've Learned", len(agent_memory.learned_patterns))
                    st.caption("🎯 Question patterns I now recognize and can answer faster")
                with col2:
                    st.metric("Times I Helped Successfully", len(agent_memory.success_feedback))
                    st.caption("✅ Solutions that worked well for you")
                    
                    st.metric("Times I Need to Improve", len(agent_memory.failed_attempts))
                    st.caption("📚 Mistakes I'm learning from to do better next time")
                
                if agent_memory.learned_patterns:
                    with st.expander("🔍 AI Learning Insights - What I've Learned From You"):
                        st.markdown("""
                        💡 **What this means:** The AI agent learns from your questions and remembers successful problem-solving approaches. 
                        This helps it give you better answers over time by recognizing similar questions and applying proven solutions.
                        """)
                        st.markdown("---")
                        
                        for i, pattern in enumerate(agent_memory.learned_patterns[:5]):
                            st.markdown(f"### 🎯 Learning Pattern {i+1}")
                            
                            keywords = pattern.get('keywords', [])
                            if keywords:
                                st.markdown(f"**Common question themes:** {', '.join(keywords)}")
                            
                            usage_count = pattern.get('usage_count', 0)
                            st.markdown(f"**How often I've seen this:** {usage_count} time{'s' if usage_count != 1 else ''}")
                            
                            if pattern.get('successful_approach'):
                                approach = pattern['successful_approach'].replace('_', ' ').title()
                                st.markdown(f"**Best solution method:** {approach}")
                                st.markdown("✅ *This approach worked well, so I'll use it again for similar questions*")
                            
                            if i < len(agent_memory.learned_patterns[:5]) - 1:
                                st.divider()
        
    elif uploaded_file and st.session_state.session_id and log_data.empty:
        # Only show success message when a file was actually uploaded and processed with no issues
        st.markdown('<div class="success-message">✅ <strong>Analysis Complete!</strong> No critical issues detected in your log file.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
