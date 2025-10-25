from typing import List, Dict, Any

# Comprehensive test cases for different capabilities
CODE_GENERATION_TESTS = [
    {
        "type": "algorithm",
        "prompt": "Write a Python implementation of a Red-Black Tree with insert and delete operations:",
        "expected_elements": ["class RedBlackTree", "def insert", "def delete", "self.color"]
    },
    {
        "type": "api_design",
        "prompt": "Create a RESTful API using FastAPI for a user management system with authentication:",
        "expected_elements": ["FastAPI", "async def", "OAuth2", "Bearer"]
    },
    {
        "type": "debugging",
        "prompt": "Debug this code and explain the fix:\ndef fibonacci(n):\n    if n <= 0: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "expected_elements": ["base case", "memoization", "stack overflow"]
    },
    {
        "type": "testing",
        "prompt": "Write unit tests for a shopping cart class that handles product addition, removal, and total calculation:",
        "expected_elements": ["class TestShoppingCart", "assertEqual", "setUp"]
    }
]

DOCUMENTATION_TESTS = [
    {
        "type": "api_docs",
        "prompt": "Write comprehensive API documentation for a payment processing endpoint:",
        "expected_elements": ["Parameters", "Returns", "Example Request", "HTTP Status Codes"]
    },
    {
        "type": "system_design",
        "prompt": "Create system architecture documentation for a microservices-based e-commerce platform:",
        "expected_elements": ["Architecture Overview", "Components", "Data Flow", "APIs"]
    },
    {
        "type": "code_comments",
        "prompt": "Document this machine learning pipeline code with detailed comments:\ndef train_model(data, params):\n    model = RandomForest()\n    model.fit(data)",
        "expected_elements": ["Parameters", "Returns", "Examples", "Notes"]
    }
]

TEACHING_TESTS = [
    {
        "type": "concept_explanation",
        "prompt": "Explain the concept of recursion to a beginner programmer with examples:",
        "expected_elements": ["base case", "recursive case", "example", "visualization"]
    },
    {
        "type": "step_by_step",
        "prompt": "Teach how to implement authentication in a web application step by step:",
        "expected_elements": ["steps", "code examples", "security considerations"]
    },
    {
        "type": "best_practices",
        "prompt": "Explain SOLID principles with real-world coding examples:",
        "expected_elements": ["Single Responsibility", "Open-Closed", "examples"]
    }
]

SYSTEM_DESIGN_TESTS = [
    {
        "type": "architecture",
        "prompt": "Design a scalable real-time chat application system:",
        "expected_elements": ["WebSocket", "Redis", "Load Balancer", "Database"]
    },
    {
        "type": "optimization",
        "prompt": "Propose optimizations for a slow-performing e-commerce website:",
        "expected_elements": ["caching", "indexing", "CDN", "database"]
    }
]

def get_all_test_cases() -> List[Dict[str, Any]]:
    """Combine all test cases with appropriate system prompts."""
    all_tests = []
    
    # Add system prompts for different roles
    for test in CODE_GENERATION_TESTS:
        test["system_prompt"] = """You are an expert programmer with deep knowledge of algorithms, 
        data structures, and software design patterns. Write clean, efficient, and well-documented code."""
        all_tests.append(test)
    
    for test in DOCUMENTATION_TESTS:
        test["system_prompt"] = """You are a technical writer with expertise in creating clear, 
        comprehensive documentation. Follow best practices for technical writing and documentation standards."""
        all_tests.append(test)
    
    for test in TEACHING_TESTS:
        test["system_prompt"] = """You are an experienced programming instructor skilled at explaining 
        complex concepts in simple terms. Provide clear explanations with relevant examples."""
        all_tests.append(test)
    
    for test in SYSTEM_DESIGN_TESTS:
        test["system_prompt"] = """You are a senior system architect with expertise in designing 
        scalable, reliable, and efficient systems. Consider all aspects including scalability, 
        reliability, and maintainability."""
        all_tests.append(test)
    
    return all_tests