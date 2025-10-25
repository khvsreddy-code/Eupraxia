import unittest
from pathlib import Path
from unified_ai import UnifiedAI, GenerationConfig
import torch
import numpy as np
from PIL import Image

class TestUnifiedAI(unittest.TestCase):
    def test_superhuman_modules(self):
        """Test all superhuman modules via UnifiedAI"""
        # Superhuman code
        code_result = self.ai.generate_superhuman_code("Write a Python function to compute Fibonacci numbers.")
        self.assertIn('code', code_result)
        self.assertTrue(len(code_result['code']) > 10)

        # Superhuman image
        image_result = self.ai.generate_superhuman_image("A futuristic cityscape at dawn.", {'height': 256, 'width': 256, 'emotion': 'awe'})
        self.assertIn('image', image_result)

        # Superhuman video
        video_result = self.ai.generate_superhuman_video("A cinematic sci-fi chase scene.", {'fps': 10, 'duration': 1, 'height': 64, 'width': 64})
        self.assertIn('frames', video_result)
        self.assertTrue(len(video_result['frames']) > 0)

        # Superhuman 3D model
        model_result = self.ai.generate_superhuman_3d_model("A hyper-realistic chair.", {'subdivisions': 2, 'radius': 1.0})
        self.assertIn('mesh', model_result)

        # Superhuman music
        music_result = self.ai.generate_superhuman_music("A symphony blending classical and jazz.")
        self.assertIn('audio', music_result)

        # Superhuman writing
        writing_result = self.ai.generate_superhuman_writing("Write a poem about AI transcendence.")
        self.assertIn('writing', writing_result)

        # Superhuman science
        science_result = self.ai.generate_superhuman_science("Invent a new field of quantum biology.")
        self.assertIn('science', science_result)

        # Superhuman engineering
        engineering_result = self.ai.generate_superhuman_engineering("Design a self-repairing bridge.")
        self.assertIn('engineering', engineering_result)

        # Superhuman business
        business_result = self.ai.generate_superhuman_business("Create a global market prediction model.")
        self.assertIn('business', business_result)

        # Superhuman art
        art_result = self.ai.generate_superhuman_art("Design a sculpture blending light and sound.")
        self.assertIn('art', art_result)

        # Superhuman health
        health_result = self.ai.generate_superhuman_health("Create a personalized longevity plan.")
        self.assertIn('health', health_result)

        # Superhuman education
        education_result = self.ai.generate_superhuman_education("Create a curriculum for learning AI ethics.")
        self.assertIn('education', education_result)

        # Superhuman meta-evolution
        meta_result = self.ai.evolve_superhuman_ai("Optimize all modules for best performance.")
        self.assertIn('evolution', meta_result)
    @classmethod
    def setUpClass(cls):
        """Initialize AI system once for all tests"""
        cls.ai = UnifiedAI()
        cls.test_dir = Path("test_outputs")
        cls.test_dir.mkdir(exist_ok=True)
        
    def test_code_generation(self):
        """Test code generation capabilities"""
        prompts = [
            "Create a simple REST API using FastAPI",
            "Implement a binary search tree in Python",
            "Create a React component for a dashboard"
        ]
        
        for i, prompt in enumerate(prompts):
            result = self.ai.generate_code(prompt)
            
            # Verify result structure
            self.assertIn('code', result)
            self.assertIn('quality_metrics', result)
            
            # Check code quality metrics
            metrics = result['quality_metrics']
            self.assertGreaterEqual(metrics['complexity'], 0)
            self.assertGreaterEqual(metrics['maintainability'], 0)
            self.assertGreaterEqual(metrics['documentation_coverage'], 0)
            
            # Save output
            output_file = self.test_dir / f"test_code_{i}.py"
            with open(output_file, 'w') as f:
                f.write(result['code'])
    
    def test_image_generation(self):
        """Test image generation capabilities"""
        prompts = [
            "A modern skyscraper at sunset",
            "A professional product photo of a smartphone",
            "A beautiful landscape with mountains and lake"
        ]
        
        config = GenerationConfig(
            image_width=512,  # Smaller for testing
            image_height=512
        )
        
        for i, prompt in enumerate(prompts):
            result = self.ai.generate_image(prompt, config)
            
            # Verify result structure
            self.assertIn('image', result)
            self.assertIn('metadata', result)
            
            # Check image properties
            image = result['image']
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.size, (512, 512))
            
            # Save output
            output_file = self.test_dir / f"test_image_{i}.png"
            image.save(output_file)
    
    def test_video_generation(self):
        """Test video generation capabilities"""
        prompt = "A cinematic city flythrough"
        config = GenerationConfig(
            duration=5,  # Short for testing
            fps=30
        )
        
        result = self.ai.generate_video(prompt, config)
        
        # Verify result structure
        self.assertIn('frames', result)
        self.assertIn('fps', result)
        self.assertIn('duration', result)
        
        # Check video properties
        frames = result['frames']
        self.assertEqual(len(frames), config.duration * config.fps)
        self.assertTrue(all(isinstance(f, np.ndarray) for f in frames))
        
        # Save output (first frame for testing)
        output_file = self.test_dir / "test_video_frame.png"
        Image.fromarray(frames[0]).save(output_file)
    
    def test_3d_model_generation(self):
        """Test 3D model generation capabilities"""
        prompt = "A modern office chair"
        config = GenerationConfig(
            resolution=128  # Lower for testing
        )
        
        result = self.ai.generate_3d_model(prompt, config)
        
        # Verify result structure
        self.assertIn('mesh', result)
        self.assertIn('metadata', result)
        
        # Check mesh properties
        mesh = result['mesh']
        self.assertTrue(len(mesh.vertices) > 0)
        self.assertTrue(len(mesh.faces) > 0)
        
        # Save output
        output_file = self.test_dir / "test_model.obj"
        mesh.export(str(output_file))
    
    def test_deep_search(self):
        """Test deep search capabilities"""
        query = "Latest advancements in quantum computing"
        result = self.ai.deep_internet_search(query)
        
        # Verify result structure
        self.assertIn('results', result)
        self.assertIn('synthesis', result)
        self.assertIn('sources', result)
        
        # Check search quality
        self.assertGreater(len(result['results']), 0)
        self.assertGreater(len(result['synthesis']), 100)  # Minimum synthesis length
    
    def test_resume_creation(self):
        """Test resume creation capabilities"""
        with open('examples/resume_info.json', 'r') as f:
            import json
            personal_info = json.load(f)
            
        result = self.ai.create_resume(personal_info, style='modern')
        
        # Verify result structure
        self.assertIn('content', result)
        self.assertIn('styling', result)
        
        # Check content quality
        self.assertIn('html', result['content'].lower())
        self.assertIn('css', result['styling'].lower())
        
        # Save output
        output_file = self.test_dir / "test_resume.html"
        with open(output_file, 'w') as f:
            html = f"""
            <html>
            <style>{result['styling']}</style>
            <body>{result['content']}</body>
            </html>
            """
            f.write(html)
    
    def test_image_editing(self):
        """Test image editing capabilities"""
        # Create a test image
        test_image = Image.new('RGB', (512, 512), 'white')
        
        instructions = "Add a blue gradient background"
        result = self.ai.edit_image(test_image, instructions)
        
        # Verify result structure
        self.assertIn('edited_image', result)
        self.assertIn('operations_applied', result)
        
        # Check image properties
        edited = result['edited_image']
        self.assertIsInstance(edited, Image.Image)
        self.assertEqual(edited.size, (512, 512))
        
        # Save output
        output_file = self.test_dir / "test_edited_image.png"
        edited.save(output_file)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test outputs"""
        import shutil
        shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main()