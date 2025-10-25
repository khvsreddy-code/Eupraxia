from .unified_ai import UnifiedAI
# Evolution Trainer for Superhuman AI System
# Implements efficient evolutionary training pipeline for continuous model improvement

import torch
from torch.cuda import amp
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import json
import gc
from PIL import Image

from backend.ai.superhuman_ai_system import SuperhumanAISystem, GenerationConfig, MetaEvolutionController


class EvolutionTrainer:

    def run_multiagent_benchmarks(self):
        """Run multi-agent, multi-modal, and meta-evolution workflows for benchmarking and invention."""
        print("[EVOLUTION] Running multi-agent, multi-modal, and meta-evolution workflows...")
        unified_ai = UnifiedAI()
        # Example: Multi-agent workflow invention
        workflow = [
            {"agent": "code", "prompt": "Write a Python function to sort a list."},
            {"agent": "image", "prompt": "Generate an icon for a sorting algorithm.", "depends_on": 0},
            {"agent": "website", "prompt": "Create a web page to demo the sorting function and icon.", "depends_on": [0,1]},
        ]
        workflow_results = unified_ai.multi_agent_pipeline(workflow)
        print("[EVOLUTION] Multi-agent workflow results:", workflow_results)

        # Meta-evolution benchmarking
        benchmark_tasks = [
            {"agent": "code", "prompt": "Write a function to compute Fibonacci numbers.", "language": "python"},
            {"agent": "code", "prompt": "Implement quicksort in JavaScript.", "language": "javascript"},
        ]
        benchmark_results = unified_ai.meta_evolution_benchmark(benchmark_tasks)
        print("[EVOLUTION] Meta-evolution benchmark results:", benchmark_results)

        # Self-competition: generate multiple outputs and select best by self-eval
        for task in benchmark_tasks:
            outputs = [unified_ai.multi_agent_pipeline([task])[0] for _ in range(3)]
            if task["agent"] == "code":
                scores = [unified_ai.superhuman_ai_coder.self_evaluate_code(o.get("code", ""), task.get("language", "python")) for o in outputs]
                best_idx = scores.index(max(scores)) if scores else 0
                print(f"[EVOLUTION] Self-competition best output for {task['prompt']}:", outputs[best_idx])
    def __init__(self, base_model: SuperhumanAISystem, training_hours: float = 1.0):
        """Initialize evolution trainer with meta-evolution and memory-efficient settings."""
        print("Initializing EvolutionTrainer...")
        self.base_model = base_model
        self.training_hours = training_hours
        self.metrics_history = []
        self.start_time = None
        self.meta_controller = MetaEvolutionController(self.base_model)
        # Training parameters
        self.learning_rate = 1e-5
        self.batch_size = 1
        print(f"EvolutionTrainer initialized for {training_hours} hours")

    def generate_synthetic_data(self, batch_size: int = 2):
        """Generate synthetic data for all domains (websites, games, assistants, etc.), including all major game builder styles."""
        data = {
            "code": [], "image": [], "video": [], "3d": [], "music": [], 
            "website": [], "game": [], "assistant": [], "voice": [], "game_wuthering": [], "game_godofwar": [], "game_godhand": [], "game_custom": []
        }
        # Website generation
        website_prompts = [
            "Create an interactive e-commerce site with 3D product views",
            "Build a social media platform with realtime features"
        ][:batch_size]
        for prompt in website_prompts:
            try:
                website = self.base_model.generate_superhuman_website(prompt)
                data["website"].append((prompt, website))
            except Exception as e:
                print(f"Website gen error: {e}")
        # Game builder: Wuthering Waves, God of War, God Hand, Custom
        try:
            # Prefer the heavy SuperhumanGameBuilder when available, otherwise use lightweight fallback
            try:
                from backend.ai.superhuman_game_builder import SuperhumanGameBuilder
                builder = SuperhumanGameBuilder()
            except Exception:
                # Lightweight fallback
                from .lightweight_generators import LightweightGameGenerator
                class _Wrapper:
                    def __init__(self):
                        self._g = LightweightGameGenerator()
                    def build_wuthering_waves_style_game(self):
                        return self._g.generate_game("Wuthering Waves style")
                    def build_god_of_war_style_game(self):
                        return self._g.generate_game("God of War style")
                    def build_god_hand_style_game(self):
                        return self._g.generate_game("God Hand style")
                    def build_custom_style_game(self, title, genre, style, style_prompt, features):
                        return self._g.generate_game(title, {"genre": genre})
                builder = _Wrapper()

            ww_game = builder.build_wuthering_waves_style_game()
            data["game_wuthering"].append(("Wuthering Waves style", ww_game))
            gow_game = builder.build_god_of_war_style_game()
            data["game_godofwar"].append(("God of War style", gow_game))
            gh_game = builder.build_god_hand_style_game()
            data["game_godhand"].append(("God Hand style", gh_game))
            custom_game = builder.build_custom_style_game(
                title="Legends of the Future",
                genre="Sci-Fi Action",
                style="Futuristic cyberpunk adventure",
                style_prompt="cyberpunk, neon, open world, advanced AI, new mechanics",
                features=["Procedural city", "AI-driven NPCs", "Dynamic missions", "Custom vehicles"]
            )
            data["game_custom"].append(("Custom style", custom_game))
        except Exception as e:
            print(f"Game builder error: {e}")
        # Game generation (legacy)
        game_prompts = [
            "Create a photorealistic open-world RPG like Wuthering Waves",
            "Build an MMO with advanced physics and combat"
        ][:batch_size]
        for prompt in game_prompts:
            try:
                game = self.base_model.generate_superhuman_game(prompt)
                data["game"].append((prompt, game))
            except Exception as e:
                print(f"Game gen error: {e}")
        # Assistant generation
        assistant_prompts = [
            "Create a natural conversational AI with human-like voice",
            "Build a domain expert AI assistant for scientific research"
        ][:batch_size]
        for prompt in assistant_prompts:
            try:
                assistant = self.base_model.generate_superhuman_assistant(prompt)
                data["assistant"].append((prompt, assistant))
            except Exception as e:
                print(f"Assistant gen error: {e}")
        
        # Code generation
        code_prompts = [
            "Create a full-stack web app",
            "Implement a battle royale game engine"
        ][:batch_size]
        for prompt in code_prompts:
            try:
                code = self.base_model.code_generator.generate(
                    **self.base_model.code_tokenizer(prompt, return_tensors="pt").to(self.base_model.device),
                    max_length=2048
                )
                data["code"].append((prompt, self.base_model.code_tokenizer.decode(code[0])))
            except Exception as e:
                print(f"Code gen error: {e}")
        # Image
        image_prompts = ["Ultra-realistic fantasy landscape", "Character concept art for AAA game"][:batch_size]
        for prompt in image_prompts:
            try:
                img = self.base_model.image_generator(prompt, num_inference_steps=20).images[0]
                data["image"].append((prompt, img))
            except Exception as e:
                print(f"Image gen error: {e}")
        # Video
        video_prompts = ["Cinematic trailer for open-world RPG", "Battle scene in futuristic city"][:batch_size]
        for prompt in video_prompts:
            try:
                vid = self.base_model.generate_video(prompt)
                data["video"].append((prompt, vid["frames"]))
            except Exception as e:
                print(f"Video gen error: {e}")
        # 3D
        model_prompts = ["3D model of a sci-fi vehicle", "Procedural terrain for game"][:batch_size]
        for prompt in model_prompts:
            try:
                model = self.base_model.model_generator.generate(
                    **self.base_model.code_tokenizer(prompt, return_tensors="pt").to(self.base_model.device),
                    max_length=1024
                )
                data["3d"].append((prompt, self.base_model.code_tokenizer.decode(model[0])))
            except Exception as e:
                print(f"3D gen error: {e}")
        # Music
        music_prompts = ["Epic orchestral game soundtrack", "Dynamic battle theme"][:batch_size]
        for prompt in music_prompts:
            try:
                music = self.base_model.music_generator.generate(
                    **self.base_model.code_tokenizer(prompt, return_tensors="pt").to(self.base_model.device),
                    max_length=1024
                )
                data["music"].append((prompt, self.base_model.code_tokenizer.decode(music[0])))
            except Exception as e:
                print(f"Music gen error: {e}")
        # Game (meta-prompt)
        game_prompts = ["Design a Genshin Impact-like open world game", "Create a PUBG-style multiplayer shooter"][:batch_size]
        for prompt in game_prompts:
            try:
                game_code = self.base_model.code_generator.generate(
                    **self.base_model.code_tokenizer(prompt, return_tensors="pt").to(self.base_model.device),
                    max_length=4096
                )
                data["game"].append((prompt, self.base_model.code_tokenizer.decode(game_code[0])))
            except Exception as e:
                print(f"Game gen error: {e}")
        return data

    def train_iteration(self):
        """Run one training iteration with meta-evolution and memory optimizations."""
        try:
            data = self.generate_synthetic_data(batch_size=self.batch_size)
            metrics = {k: 0.0 for k in data.keys()}
            metrics["memory_usage"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            # Evaluate quality for each domain
            for domain, samples in data.items():
                for prompt, output in samples:
                    metrics[domain] += self.evaluate_quality(output)
            # Meta-evolution controller hook
            self.meta_controller.monitor_and_evolve()
            self.metrics_history.append(metrics)
            torch.cuda.empty_cache()
            return metrics
        except Exception as e:
            print(f"Training iteration error: {str(e)}")
            return None

    def evaluate_quality(self, output):
        """Evaluate quality for any output type (text, image, code, etc.)"""
        try:
            if isinstance(output, Image.Image):
                arr = np.array(output)
                brightness = np.mean(arr)
                contrast = np.std(arr)
                score = (brightness / 255.0) * 0.3 + (contrast / 128.0) * 0.7
                return min(max(score, 0.0), 1.0)
            elif isinstance(output, str):
                # Text/code: length and diversity
                length_score = min(len(output) / 1000.0, 1.0)
                diversity_score = len(set(output.split())) / (len(output.split()) + 1e-5)
                return 0.5 * length_score + 0.5 * diversity_score
            elif isinstance(output, list):
                # Video/frames: average image quality
                if output and isinstance(output[0], Image.Image):
                    return np.mean([self.evaluate_quality(f) for f in output])
                return 0.0
            return 0.0
        except Exception as e:
            print(f"Error evaluating quality: {str(e)}")
            return 0.0

    def save_progress(self):
        """Save evolution progress and metrics."""
        try:
            save_dir = Path("evolution_checkpoints")
            save_dir.mkdir(exist_ok=True)
            with open(save_dir / "metrics_history.json", "w") as f:
                json.dump(self.metrics_history, f)
        except Exception as e:
            print(f"Error saving progress: {str(e)}")

    def evolve(self):
        """Run evolution process for specified duration."""
        print(f"\nStarting evolution process for {self.training_hours} hours...")
        print("Starting AI evolution process...")
        self.start_time = time.time()
        self.metrics_history = []
        

        try:
            while (time.time() - self.start_time) < (self.training_hours * 3600):
                # Run training iteration
                metrics = self.train_iteration()
                if metrics:
                    self.metrics_history.append(metrics)
                    print(f"\rIteration {len(self.metrics_history)}: "
                          + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() if k != 'memory_usage')
                          + f" | Memory: {metrics['memory_usage']:.2f}GB",
                          end="", flush=True)
                # Save periodically
                if len(self.metrics_history) % 10 == 0:
                    Path("evolution_checkpoints").mkdir(exist_ok=True)
                    with open("evolution_checkpoints/metrics_history.json", "w") as f:
                        json.dump(self.metrics_history, f)
                time.sleep(0.1)
            print("\nEvolution complete!")
        except KeyboardInterrupt:
            print("\nEvolution interrupted!")
        except Exception as e:
            print(f"\nEvolution error: {str(e)}")
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    # Initialize UnifiedAI as the base model (uses lightweight fallback on CPU)
    base_model = UnifiedAI()
    # Create and run evolution trainer for 1.5 hours
    trainer = EvolutionTrainer(base_model, training_hours=1.5)
    trainer.evolve()
    # Run multi-agent, multi-modal, and meta-evolution benchmarks after evolution
    trainer.run_multiagent_benchmarks()