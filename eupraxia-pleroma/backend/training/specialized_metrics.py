import re
from typing import List, Dict, Any
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from dataclasses import dataclass

@dataclass
class CodeQualityMetrics:
    complexity: float
    readability: float
    completeness: float
    correctness: float
    documentation: float

@dataclass
class DocumentationQualityMetrics:
    completeness: float
    clarity: float
    structure: float
    examples_quality: float

@dataclass
class TeachingQualityMetrics:
    clarity: float
    depth: float
    examples_quality: float
    progression: float

class SpecializedMetrics:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_code_quality(self, code: str) -> CodeQualityMetrics:
        """Evaluate the quality of generated code."""
        # Measure code complexity
        complexity = self._measure_code_complexity(code)
        
        # Check code readability
        readability = self._measure_code_readability(code)
        
        # Assess completeness
        completeness = self._assess_code_completeness(code)
        
        # Check correctness (syntax and basic logic)
        correctness = self._check_code_correctness(code)
        
        # Evaluate documentation
        documentation = self._evaluate_code_documentation(code)
        
        return CodeQualityMetrics(
            complexity=complexity,
            readability=readability,
            completeness=completeness,
            correctness=correctness,
            documentation=documentation
        )
    
    def evaluate_documentation_quality(self, doc: str) -> DocumentationQualityMetrics:
        """Evaluate the quality of generated documentation."""
        # Check completeness
        completeness = self._measure_doc_completeness(doc)
        
        # Assess clarity
        clarity = self._measure_doc_clarity(doc)
        
        # Evaluate structure
        structure = self._evaluate_doc_structure(doc)
        
        # Check examples
        examples_quality = self._assess_doc_examples(doc)
        
        return DocumentationQualityMetrics(
            completeness=completeness,
            clarity=clarity,
            structure=structure,
            examples_quality=examples_quality
        )
    
    def evaluate_teaching_quality(self, content: str) -> TeachingQualityMetrics:
        """Evaluate the quality of teaching content."""
        # Measure clarity of explanation
        clarity = self._measure_teaching_clarity(content)
        
        # Assess depth of coverage
        depth = self._measure_teaching_depth(content)
        
        # Evaluate examples
        examples_quality = self._evaluate_teaching_examples(content)
        
        # Check logical progression
        progression = self._assess_teaching_progression(content)
        
        return TeachingQualityMetrics(
            clarity=clarity,
            depth=depth,
            examples_quality=examples_quality,
            progression=progression
        )
    
    def _measure_code_complexity(self, code: str) -> float:
        """Measure cyclomatic complexity and nesting depth."""
        # Count control structures
        control_structures = len(re.findall(r'\b(if|for|while|def|class)\b', code))
        # Count nesting levels
        max_nesting = 0
        current_nesting = 0
        for line in code.split('\n'):
            indent = len(line) - len(line.lstrip())
            current_nesting = indent // 4  # Assuming 4-space indentation
            max_nesting = max(max_nesting, current_nesting)
        
        # Normalize to 0-1 scale (lower is better)
        complexity = 1 - (0.7 ** (control_structures / 10 + max_nesting / 5))
        return max(0, min(1, complexity))
    
    def _measure_code_readability(self, code: str) -> float:
        """Assess code readability based on various factors."""
        factors = {
            'line_length': self._check_line_length(code),
            'naming': self._check_naming_conventions(code),
            'spacing': self._check_spacing(code),
            'comments': self._check_comments_ratio(code)
        }
        return np.mean(list(factors.values()))
    
    def _assess_code_completeness(self, code: str) -> float:
        """Check if code has all necessary components."""
        required_elements = {
            'imports': bool(re.search(r'^import\s+|^from\s+.*\s+import\s+', code, re.M)),
            'function_def': bool(re.search(r'\bdef\s+\w+\s*\(', code)),
            'return': bool(re.search(r'\breturn\b', code)),
            'error_handling': bool(re.search(r'\btry\b.*\bexcept\b', code, re.S))
        }
        return sum(required_elements.values()) / len(required_elements)
    
    def _check_code_correctness(self, code: str) -> float:
        """Basic syntax and logic checking."""
        try:
            compile(code, '<string>', 'exec')
            syntax_score = 1.0
        except SyntaxError:
            syntax_score = 0.0
        
        # Check basic logic patterns
        logic_patterns = {
            'initialization': bool(re.search(r'=\s*\w+', code)),
            'conditions': bool(re.search(r'if\s+.*:', code)),
            'loops': bool(re.search(r'(for|while)\s+.*:', code))
        }
        logic_score = sum(logic_patterns.values()) / max(1, len(logic_patterns))
        
        return (syntax_score + logic_score) / 2
    
    def _evaluate_code_documentation(self, code: str) -> float:
        """Evaluate code documentation quality."""
        doc_patterns = {
            'module_doc': bool(re.search(r'^""".*?"""', code, re.S)),
            'function_doc': bool(re.search(r'def.*?""".*?"""', code, re.S)),
            'inline_comments': bool(re.search(r'#.*\w+', code))
        }
        return sum(doc_patterns.values()) / len(doc_patterns)
    
    def _measure_doc_completeness(self, doc: str) -> float:
        """Measure documentation completeness."""
        required_sections = {
            'overview': bool(re.search(r'overview|introduction|description', doc, re.I)),
            'parameters': bool(re.search(r'parameters|args|arguments', doc, re.I)),
            'returns': bool(re.search(r'returns|output', doc, re.I)),
            'examples': bool(re.search(r'examples?|usage', doc, re.I))
        }
        return sum(required_sections.values()) / len(required_sections)
    
    def _measure_doc_clarity(self, doc: str) -> float:
        """Assess documentation clarity."""
        factors = {
            'sentence_length': self._check_sentence_length(doc),
            'technical_terms': self._check_technical_terms(doc),
            'structure': self._check_doc_structure(doc)
        }
        return np.mean(list(factors.values()))
    
    def _evaluate_doc_structure(self, doc: str) -> float:
        """Evaluate documentation structure."""
        structure_elements = {
            'sections': bool(re.search(r'^#+\s+\w+', doc, re.M)),
            'lists': bool(re.search(r'^\s*[-*]\s+\w+', doc, re.M)),
            'code_blocks': bool(re.search(r'```\w*\n.*?\n```', doc, re.S))
        }
        return sum(structure_elements.values()) / len(structure_elements)
    
    def _assess_doc_examples(self, doc: str) -> float:
        """Assess quality of documentation examples."""
        example_patterns = {
            'code_examples': len(re.findall(r'```\w*\n.*?\n```', doc, re.S)),
            'inline_code': len(re.findall(r'`[^`]+`', doc)),
            'usage_examples': bool(re.search(r'example usage|how to use', doc, re.I))
        }
        # Normalize to 0-1
        return min(1.0, (example_patterns['code_examples'] / 2 + 
                        example_patterns['inline_code'] / 5 +
                        example_patterns['usage_examples']) / 3)
    
    # Helper methods for code evaluation
    def _check_line_length(self, code: str) -> float:
        lines = [line for line in code.split('\n') if line.strip()]
        if not lines:
            return 0.0
        long_lines = sum(1 for line in lines if len(line) > 80)
        return 1 - (long_lines / len(lines))
    
    def _check_naming_conventions(self, code: str) -> float:
        patterns = {
            'snake_case': r'\b[a-z][a-z0-9_]*\b',
            'camel_case': r'\b[A-Z][a-zA-Z0-9]*\b',
            'constants': r'\b[A-Z][A-Z0-9_]*\b'
        }
        scores = []
        for pattern in patterns.values():
            matches = re.findall(pattern, code)
            scores.append(bool(matches))
        return sum(scores) / len(scores)
    
    def _check_spacing(self, code: str) -> float:
        spacing_patterns = {
            'after_comma': r',\s+\w',
            'around_operators': r'\w\s*[+\-*/]\s*\w',
            'indentation': r'^\s{4}\w'
        }
        scores = []
        for pattern in spacing_patterns.values():
            matches = re.findall(pattern, code, re.M)
            scores.append(bool(matches))
        return sum(scores) / len(scores)
    
    def _check_comments_ratio(self, code: str) -> float:
        code_lines = [line for line in code.split('\n') if line.strip()]
        comment_lines = [line for line in code_lines if re.match(r'\s*#', line)]
        if not code_lines:
            return 0.0
        ratio = len(comment_lines) / len(code_lines)
        return min(1.0, ratio * 2)  # Normalize to 0-1, optimal ratio around 0.5
    
    # Teaching evaluation helper methods
    def _measure_teaching_clarity(self, content: str) -> float:
        factors = {
            'simple_language': self._check_language_complexity(content),
            'examples': bool(re.search(r'(example|instance|case):', content, re.I)),
            'explanations': bool(re.search(r'(because|therefore|thus)', content, re.I))
        }
        return np.mean(list(factors.values()))
    
    def _measure_teaching_depth(self, content: str) -> float:
        depth_indicators = {
            'concepts': len(re.findall(r'\b(concept|principle|theory)\b', content, re.I)),
            'technical_terms': len(re.findall(r'\b[A-Z][a-zA-Z]+\b', content)),
            'explanations': len(re.findall(r'\b(because|therefore|thus)\b', content, re.I))
        }
        return min(1.0, sum(depth_indicators.values()) / 15)  # Normalize to 0-1
    
    def _evaluate_teaching_examples(self, content: str) -> float:
        example_features = {
            'code_examples': bool(re.search(r'```.*?```', content, re.S)),
            'practical_cases': bool(re.search(r'(real[- ]world|practical) example', content, re.I)),
            'step_by_step': bool(re.search(r'step [0-9]|first|second|finally', content, re.I))
        }
        return sum(example_features.values()) / len(example_features)
    
    def _assess_teaching_progression(self, content: str) -> float:
        progression_markers = {
            'sequential': bool(re.search(r'first|second|then|finally', content, re.I)),
            'building_up': bool(re.search(r'basic|intermediate|advanced', content, re.I)),
            'connections': bool(re.search(r'related|connected|similarly', content, re.I))
        }
        return sum(progression_markers.values()) / len(progression_markers)
    
    def _check_language_complexity(self, content: str) -> float:
        words = content.split()
        if not words:
            return 0.0
        complex_words = sum(1 for word in words if len(word) > 12)
        return 1 - (complex_words / len(words))