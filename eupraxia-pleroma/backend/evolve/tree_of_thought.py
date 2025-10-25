"""
Tree of Thought Reasoning System
Implements advanced multi-path reasoning with parallel hypothesis exploration
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class ThoughtNode:
    thought: str
    score: float
    children: List['ThoughtNode']
    parent: Optional['ThoughtNode']
    depth: int
    state: Dict

class TreeOfThoughtReasoner:
    def __init__(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_depth: int = 5,
        beam_width: int = 3,
        temperature: float = 0.7
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.temperature = temperature

    def evaluate_thought(self, thought: str, state: Dict) -> float:
        """Evaluate the quality of a thought based on coherence and relevance"""
        prompt = f"Evaluate this reasoning step:\n{thought}\n\nScore (0-1):"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.2,
                num_return_sequences=1
            )
        score_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            score = float(score_text.split()[-1])
            return min(max(score, 0), 1)
        except:
            return 0.5

    def generate_thoughts(self, 
                        state: Dict,
                        context: str,
                        n: int = 3) -> List[str]:
        """Generate multiple potential thoughts given the current state"""
        prompt = (
            f"Context: {context}\n"
            f"Current state: {state}\n"
            "Generate {n} different reasoning steps:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                temperature=self.temperature,
                num_return_sequences=n,
                do_sample=True
            )
        thoughts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        return thoughts

    def expand_node(self, 
                   node: ThoughtNode,
                   context: str) -> List[ThoughtNode]:
        """Expand a node by generating and evaluating new thoughts"""
        if node.depth >= self.max_depth:
            return []
        
        thoughts = self.generate_thoughts(node.state, context)
        child_nodes = []
        
        for thought in thoughts:
            new_state = dict(node.state)
            new_state['last_thought'] = thought
            score = self.evaluate_thought(thought, new_state)
            
            child = ThoughtNode(
                thought=thought,
                score=score,
                children=[],
                parent=node,
                depth=node.depth + 1,
                state=new_state
            )
            child_nodes.append(child)
            
        return sorted(child_nodes, key=lambda x: x.score, reverse=True)[:self.beam_width]

    def search(self, 
              initial_state: Dict,
              context: str) -> List[ThoughtNode]:
        """Perform beam search through the tree of thoughts"""
        root = ThoughtNode(
            thought="Initial state",
            score=1.0,
            children=[],
            parent=None,
            depth=0,
            state=initial_state
        )
        
        frontier = [root]
        best_paths = []
        
        for _ in range(self.max_depth):
            next_frontier = []
            for node in frontier:
                children = self.expand_node(node, context)
                node.children.extend(children)
                next_frontier.extend(children)
                
            if not next_frontier:
                break
                
            frontier = sorted(next_frontier, key=lambda x: x.score, reverse=True)[:self.beam_width]
            best_paths = frontier
            
        return best_paths

    def get_best_reasoning_path(self, 
                              initial_state: Dict,
                              context: str) -> List[str]:
        """Get the highest-scoring reasoning path"""
        best_paths = self.search(initial_state, context)
        if not best_paths:
            return []
            
        best_node = max(best_paths, key=lambda x: x.score)
        path = []
        current = best_node
        
        while current.parent is not None:
            path.append(current.thought)
            current = current.parent
            
        return list(reversed(path))