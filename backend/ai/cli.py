import click
from unified_ai import UnifiedAI, GenerationConfig
from pathlib import Path
import json
from PIL import Image
import logging
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

@click.group()
def cli():
    """Unified AI CLI - Generate anything from text prompts"""
    pass

@cli.command()
@click.argument('prompt')
@click.option('--output', '-o', type=str, default='output.py', help='Output file for generated code')
@click.option('--temperature', '-t', type=float, default=0.7, help='Temperature for generation')
def generate_code(prompt, output, temperature):
    """Generate high-quality code from text prompt"""
    with console.status("[bold green]Generating code...") as status:
        ai = UnifiedAI()
        config = GenerationConfig(temperature=temperature)
        
        result = ai.generate_code(prompt, config)
        
        # Save code
        with open(output, 'w') as f:
            f.write(result['code'])
            
        # Display results
        console.print("\n[bold green]Generated Code:[/bold green]")
        syntax = Syntax(result['code'], "python", theme="monokai", line_numbers=True)
        console.print(syntax)
        
        console.print("\n[bold blue]Code Quality Metrics:[/bold blue]")
        for metric, value in result['quality_metrics'].items():
            console.print(f"{metric}: {value:.2%}")

@cli.command()
@click.argument('prompt')
@click.option('--output', '-o', type=str, default='output.png', help='Output file for generated image')
@click.option('--width', '-w', type=int, default=1024, help='Image width')
@click.option('--height', '-h', type=int, default=1024, help='Image height')
def generate_image(prompt, output, width, height):
    """Generate high-quality image from text prompt"""
    with console.status("[bold green]Generating image...") as status:
        ai = UnifiedAI()
        config = GenerationConfig(image_width=width, image_height=height)
        
        result = ai.generate_image(prompt, config)
        result['image'].save(output)
        
        console.print(f"\n[bold green]Image saved to: {output}")
        console.print("\n[bold blue]Image Metadata:[/bold blue]")
        for key, value in result['metadata'].items():
            console.print(f"{key}: {value}")

@cli.command()
@click.argument('prompt')
@click.option('--output', '-o', type=str, default='output.mp4', help='Output file for generated video')
@click.option('--duration', '-d', type=int, default=10, help='Video duration in seconds')
@click.option('--fps', '-f', type=int, default=30, help='Frames per second')
def generate_video(prompt, output, duration, fps):
    """Generate high-quality video from text prompt"""
    with console.status("[bold green]Generating video...") as status:
        ai = UnifiedAI()
        config = GenerationConfig(duration=duration, fps=fps)
        
        result = ai.generate_video(prompt, config)
        
        # Save video frames
        import cv2
        import numpy as np
        
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                            (result['frames'][0].shape[1], result['frames'][0].shape[0]))
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Saving video...", total=len(result['frames']))
            
            for frame in result['frames']:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                progress.advance(task)
                
        out.release()
        
        console.print(f"\n[bold green]Video saved to: {output}")
        console.print("\n[bold blue]Video Info:[/bold blue]")
        console.print(f"Duration: {duration}s")
        console.print(f"FPS: {fps}")

@cli.command()
@click.argument('prompt')
@click.option('--output', '-o', type=str, default='output.obj', help='Output file for generated 3D model')
@click.option('--resolution', '-r', type=int, default=256, help='Model resolution')
def generate_3d(prompt, output, resolution):
    """Generate high-quality 3D model from text prompt"""
    with console.status("[bold green]Generating 3D model...") as status:
        ai = UnifiedAI()
        config = GenerationConfig(resolution=resolution)
        
        result = ai.generate_3d_model(prompt, config)
        
        # Save 3D model
        result['mesh'].export(output)
        
        console.print(f"\n[bold green]3D model saved to: {output}")
        console.print("\n[bold blue]Model Statistics:[/bold blue]")
        for key, value in result['metadata'].items():
            console.print(f"{key}: {value}")

@cli.command()
@click.argument('query')
@click.option('--output', '-o', type=str, default='search_results.json', help='Output file for search results')
@click.option('--depth', '-d', type=int, default=3, help='Search depth')
def deep_search(query, output, depth):
    """Perform deep internet search and analysis"""
    with console.status("[bold green]Performing deep search...") as status:
        ai = UnifiedAI()
        result = ai.deep_internet_search(query, depth)
        
        # Save results
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
            
        console.print(f"\n[bold green]Search results saved to: {output}")
        console.print(f"\nFound {result['sources']} sources")
        
        console.print("\n[bold blue]Analysis Summary:[/bold blue]")
        console.print(Panel(result['synthesis'], title="Search Synthesis"))

@cli.command()
@click.argument('info_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=str, default='resume.html', help='Output file for resume')
@click.option('--style', '-s', type=str, default='professional', help='Resume style')
def create_resume(info_file, output, style):
    """Create a professional resume"""
    with console.status("[bold green]Creating resume...") as status:
        # Load personal info
        with open(info_file) as f:
            personal_info = json.load(f)
            
        ai = UnifiedAI()
        result = ai.create_resume(personal_info, style)
        
        # Save resume
        with open(output, 'w') as f:
            html = f"""
            <html>
            <style>{result['styling']}</style>
            <body>{result['content']}</body>
            </html>
            """
            f.write(html)
            
        console.print(f"\n[bold green]Resume saved to: {output}")

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('instructions')
@click.option('--output', '-o', type=str, default='edited_image.png', help='Output file for edited image')
def edit_image(image_path, instructions, output):
    """Edit image based on instructions"""
    with console.status("[bold green]Editing image...") as status:
        ai = UnifiedAI()
        
        # Load image
        image = Image.open(image_path)
        
        result = ai.edit_image(image, instructions)
        result['edited_image'].save(output)
        
        console.print(f"\n[bold green]Edited image saved to: {output}")
        console.print("\n[bold blue]Applied Operations:[/bold blue]")
        for op in result['operations_applied']:
            console.print(f"- {op}")

if __name__ == '__main__':
    cli()