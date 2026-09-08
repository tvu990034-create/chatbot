"""
main.py – local-chatbot entry point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Provides a Typer CLI with three sub-commands:

  python main.py ui        – launch the Gradio chat interface (default)
  python main.py api       – launch the FastAPI REST server
  python main.py both      – launch API + UI concurrently
  python main.py ingest    – ingest documents into the RAG index
  python main.py chat      – quick single-turn chat from the terminal
  python main.py status    – show config summary and backend health
  python main.py benchmark – run hardware benchmark and model recommendations
  python main.py finetune  – fine-tune model with LoRA/QLoRA
  python main.py adapters  – manage LoRA adapters
  python main.py merge     – merge multiple models with TIES, DARE, SLERP, etc.
  python main.py advanced-benchmark – run enhanced MMLU-Pro benchmark with advanced reasoning
  python main.py turbo chat      – fast + smart one-shot chat (optimizer stack)
  python main.py turbo run       – interactive optimizer REPL
  python main.py turbo check     – verify all optimization modules are working
  python main.py turbo stats     – show live optimizer telemetry
  python main.py turbo bench     – optimized vs. baseline timing comparison

Usage examples
--------------
  # Start everything (API on :8000, UI on :7860)
  python main.py both

  # UI only with a public Gradio share link
  python main.py ui --share

  # Ingest a folder of PDFs
  python main.py ingest --path ./my-docs

  # One-shot terminal chat
  python main.py chat "What is RAG?"

  # Fast + smart one-shot chat through the optimizer stack
  python main.py turbo chat "What is RAG?" --baseline
  python main.py turbo run

  # Check what is running
  python main.py status

  # Run hardware benchmark
  python main.py benchmark
  python main.py benchmark --gpu "RTX 4090" --top 3 --speed fast

  # Fine-tune model with LoRA
  python main.py finetune --method qlora --dataset gsm8k --max-samples 1000
  python main.py finetune --recipe gsm8k --output-dir ./my-math-adapter

  # Manage LoRA adapters
  python main.py adapters list
  python main.py adapters load --adapter my-adapter

  # Merge models
  python main.py merge --method ties --models model1 model2
  python main.py merge --method dare --models model1 model2 --output ./merged
  python main.py merge --method task_arithmetic --models math_adapter code_adapter

  # Run enhanced MMLU-Pro benchmark (automatic advanced reasoning by default)
  python main.py advanced-benchmark --model ollama/llama3 --num-questions 5
  python main.py advanced-benchmark --categories math chemistry --difficulty hard
  python main.py advanced-benchmark --offline  # Run without Ollama
  python main.py advanced-benchmark --difficulty mixed  # Test all difficulty levels
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Make sure the project root is on sys.path when run as a script
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings

from optimizer_cli import app as turbo_app

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("local-chatbot")
console = Console()

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="local-chatbot",
    help="Local AI chatbot powered by vLLM/SGLang, LiteLLM, LangGraph, LlamaIndex, Haystack & Aider.",
    add_completion=False,
)

app.add_typer(turbo_app, name="turbo",
              help="Make the local AI faster and smarter (optimizer-powered commands).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_api_server(host: str, port: int, reload: bool) -> None:
    """Run the FastAPI server (blocking)."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else settings.api_workers,
        log_level=settings.log_level.lower(),
    )


def _start_gradio_ui(share: bool) -> None:
    """Run the Gradio UI (blocking)."""
    from ui.gradio_ui import launch
    launch(share=share)


def _print_banner() -> None:
    console.print(Panel.fit(
        f"[bold cyan]{settings.app_name}[/bold cyan]  v{settings.app_version}\n"
        "[dim]vLLM · SGLang · LiteLLM · LangGraph · LlamaIndex · Haystack · Aider[/dim]",
        border_style="cyan",
    ))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def ui(
    host: str  = typer.Option("0.0.0.0", "--host"),
    port: int  = typer.Option(8080, "--port"),
) -> None:
    """Launch the Lovable React UI."""
    _print_banner()
    console.print(f"[green]Lovable React UI[/green]")
    console.print(f"[dim]To start UI manually:[/dim]")
    console.print(f"  cd react-ui")
    console.print(f"  npm run dev -- --host {host} --port {port}")
    console.print(f"[dim]Then access at: http://{host}:{port}[/dim]")
    console.print("[yellow]Note: UI must be started manually due to npm subprocess limitations[/yellow]")


@app.command()
def api(
    host:   str  = typer.Option(settings.api_host,   "--host"),
    port:   int  = typer.Option(settings.api_port,   "--port"),
    reload: bool = typer.Option(settings.api_reload, "--reload/--no-reload",
                                help="Auto-reload on code changes (dev mode)"),
) -> None:
    """Launch the FastAPI REST server."""
    _print_banner()
    console.print(f"[green]Starting FastAPI on http://{host}:{port}[/green]")
    console.print(f"[dim]Docs: http://{host}:{port}/docs[/dim]")
    _start_api_server(host=host, port=port, reload=reload)


@app.command()
def both(
    api_host:   str  = typer.Option(settings.api_host, "--api-host"),
    api_port:   int  = typer.Option(settings.api_port, "--api-port"),
    ui_host:    str  = typer.Option(settings.ui_host,  "--ui-host"),
    ui_port:    int  = typer.Option(settings.ui_port,  "--ui-port"),
    share:      bool = typer.Option(False, "--share"),
    reload:     bool = typer.Option(False, "--reload/--no-reload"),
) -> None:
    """Launch FastAPI + Gradio UI concurrently (two processes)."""
    _print_banner()
    console.print(
        f"[green]API[/green]  → http://{api_host}:{api_port}\n"
        f"[green]UI[/green]   → http://{ui_host}:{ui_port}"
    )

    api_proc = multiprocessing.Process(
        target=_start_api_server,
        args=(api_host, api_port, reload),
        daemon=True,
    )
    ui_proc = multiprocessing.Process(
        target=_start_gradio_ui,
        args=(share,),
        daemon=True,
    )

    api_proc.start()
    ui_proc.start()

    try:
        api_proc.join()
        ui_proc.join()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down …[/yellow]")
        api_proc.terminate()
        ui_proc.terminate()


@app.command()
def ingest(
    path: Optional[Path] = typer.Option(
        None, "--path", "-p",
        help="Directory or file to ingest. Defaults to data/docs/.",
    ),
    rebuild: bool = typer.Option(False, "--rebuild", help="Force full index rebuild"),
) -> None:
    """Ingest documents into the RAG vector store."""
    _print_banner()
    from config import RAGProvider

    target = path or settings.rag_docs_dir
    console.print(f"Ingesting from: [bold]{target}[/bold]")

    file_paths: list[Path] = []
    if Path(target).is_file():
        file_paths = [Path(target)]
    elif Path(target).is_dir():
        file_paths = list(Path(target).rglob("*"))
        file_paths = [p for p in file_paths if p.is_file()]
    else:
        console.print(f"[red]Path not found: {target}[/red]")
        raise typer.Exit(1)

    if settings.rag_provider in (RAGProvider.LLAMA_INDEX, RAGProvider.BOTH):
        console.print("[cyan]Ingesting into LlamaIndex …[/cyan]")
        try:
            from rag.llama_index_rag import LlamaIndexRAG
            rag = LlamaIndexRAG()
            if rebuild:
                rag.build_index(force_rebuild=True)
            else:
                n = rag.add_documents(file_paths)
                console.print(f"[green]LlamaIndex: {n} node(s) added.[/green]")
        except Exception as exc:
            console.print(f"[red]LlamaIndex ingest failed: {exc}[/red]")

    if settings.rag_provider in (RAGProvider.HAYSTACK, RAGProvider.BOTH):
        console.print("[cyan]Ingesting into Haystack …[/cyan]")
        try:
            from rag.haystack_pipeline import HaystackRAG
            hs = HaystackRAG()
            hs.build_pipeline()
            n = hs.ingest(file_paths)
            console.print(f"[green]Haystack: {n} document(s) stored.[/green]")
        except Exception as exc:
            console.print(f"[red]Haystack ingest failed: {exc}[/red]")


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send to the chatbot"),
    model:   Optional[str] = typer.Option(None, "--model", "-m"),
    no_rag:  bool = typer.Option(False, "--no-rag",   help="Disable RAG retrieval"),
    no_agent:bool = typer.Option(False, "--no-agent", help="Skip LangGraph agent"),
    stream:  bool = typer.Option(True,  "--stream/--no-stream"),
) -> None:
    """Send a single message to the chatbot and print the reply."""
    _print_banner()

    if no_agent:
        # Direct LiteLLM call
        from gateway.litellm_gateway import chat_stream, chat as gw_chat
        msgs = [{"role": "user", "content": message}]
        if stream:
            console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
            for chunk in chat_stream(msgs, model=model or None):
                console.print(chunk, end="", highlight=False)
            console.print()
        else:
            reply = gw_chat(msgs, model=model or None)
            console.print(f"[bold cyan]Assistant:[/bold cyan] {reply}")
    else:
        from agents.langgraph_agent import chat as agent_chat
        console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
        reply = agent_chat(message)
        console.print(reply)


@app.command()
def status() -> None:
    """Show configuration summary and backend health."""
    _print_banner()

    table = Table(title="Configuration", show_header=False, border_style="dim")
    table.add_column("Key",   style="cyan",  no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Default model",    settings.default_model)
    table.add_row("Local backend",    settings.local_backend.value)
    table.add_row("Local model",      settings.local_model_name)
    table.add_row("RAG provider",     settings.rag_provider.value)
    table.add_row("Embedding model",  settings.embedding_model)
    table.add_row("Docs directory",   str(settings.rag_docs_dir))
    table.add_row("API endpoint",     f"http://{settings.api_host}:{settings.api_port}")
    table.add_row("UI endpoint",      f"http://{settings.ui_host}:{settings.ui_port}")
    table.add_row("Aider repo",       str(settings.aider_repo_path))
    table.add_row("Aider read-only",  str(settings.aider_read_only))

    console.print(table)

    # Backend health
    try:
        from backends.model_server import get_backend_status
        s = get_backend_status()
        icon = "[OK] healthy" if s.get("healthy") else "[FAIL] unreachable"
        console.print(f"\nBackend [{s['backend'].upper()}]: {icon}")
    except Exception as exc:
        console.print(f"\n[yellow]Backend status unavailable: {exc}[/yellow]")


@app.command()
def benchmark(
    gpu_name: Optional[str] = typer.Option(None, "--gpu", help="Simulate specific GPU (e.g., 'RTX 4090')"),
    top: int = typer.Option(5, "--top", "-n", help="Number of top models to show"),
    speed: str = typer.Option("any", "--speed", help="Speed filter: any, usable, fast"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run hardware benchmark and show model recommendations."""
    from gateway.hardware_benchmark import get_hardware_benchmark
    
    benchmark = get_hardware_benchmark()
    
    # Get recommendations
    if gpu_name:
        models = benchmark.get_gpu_recommendations(gpu_name=gpu_name, top_n=top, speed_filter=speed)
    else:
        models = benchmark.get_cpu_recommendations(top_n=top)
    
    if json_output:
        import json
        hardware_info = benchmark.get_hardware_info()
        output = {
            "hardware": hardware_info,
            "simulated_gpu": gpu_name,
            "recommendations": models
        }
        print(json.dumps(output, indent=2))
    else:
        # Simple text output to avoid Rich console issues
        print("Hardware Benchmark Results")
        print("=" * 40)
        
        hardware_info = benchmark.get_hardware_info()
        if hardware_info.get("available"):
            cpu = hardware_info.get("cpu", "Unknown")
            ram_gb = hardware_info.get("ram_bytes", 0) / (1024**3)
            gpus = hardware_info.get("gpus", [])
            
            print(f"CPU: {cpu}")
            print(f"RAM: {ram_gb:.1f} GB")
            if gpus:
                for gpu_info in gpus:
                    vram_gb = gpu_info.get("vram_bytes", 0) / (1024**3)
                    print(f"GPU: {gpu_info.get('name', 'Unknown')} ({vram_gb:.1f} GB)")
            else:
                print("GPU: None detected")
        else:
            print("Hardware detection not available (whichllm not found)")
            print("Install with: pip install whichllm")
            return
        
        print()
        
        if gpu_name:
            print(f"Simulating GPU: {gpu_name}")
        else:
            print("CPU-only recommendations")
        
        print()
        
        if models:
            for i, model in enumerate(models, 1):
                model_id = model.get("model_id", "unknown")
                params = model.get("parameter_count", 0) / 1e9
                quant = model.get("quant_type", "unknown")
                speed = model.get("estimated_tok_per_sec", 0)
                score = model.get("quality_score", 0)
                fit_type = model.get("fit_type", "unknown")
                
                print(f"{i}. {model_id}")
                print(f"   Parameters: {params:.1f}B | Quant: {quant}")
                print(f"   Speed: {speed:.1f} tok/s | Quality Score: {score:.1f}")
                print(f"   Fit: {fit_type}")
                print()
        else:
            print("No recommendations available")


@app.command()
def finetune(
    method: str = typer.Option("qlora", "--method", "-m", help="Fine-tuning method: lora, qlora, ada_lora, dora"),
    dataset: str = typer.Option("gsm8k", "--dataset", "-d", help="Dataset name or path"),
    recipe: Optional[str] = typer.Option(None, "--recipe", "-r", help="Math recipe: gsm8k, math, multi_math, process_supervision"),
    model: Optional[str] = typer.Option(None, "--model", help="Base model name"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for adapters"),
    max_samples: int = typer.Option(1000, "--max-samples", "-n", help="Maximum training samples"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(2e-4, "--lr", help="Learning rate"),
) -> None:
    """Fine-tune model with LoRA/QLoRA."""
    _print_banner()
    
    try:
        from finetuning.lora_trainer import get_trainer
        from finetuning.dataset_processor import get_dataset_processor
        from finetuning.math_recipes import get_math_recipe
    except ImportError as e:
        console.print(f"[red]Fine-tuning dependencies not installed: {e}[/red]")
        console.print("[yellow]Install with: pip install peft datasets tensorboard[/yellow]")
        raise typer.Exit(1)
    
    model_name = model or settings.local_model_name
    output_path = output_dir or settings.finetuning_output_dir
    
    console.print(f"[green]Fine-tuning model: {model_name}[/green]")
    console.print(f"[dim]Method: {method} | Dataset: {dataset}[/dim]")
    
    # Use recipe if specified
    if recipe:
        console.print(f"[cyan]Using math recipe: {recipe}[/cyan]")
        try:
            recipe_instance = get_math_recipe(recipe, model_name, output_path, method)
            train_dataset = recipe_instance.prepare_training(max_samples)
            recipe_instance.train(train_dataset)
            console.print(f"[green]Fine-tuning completed! Adapter saved to {output_path}[/green]")
            return
        except Exception as e:
            console.print(f"[red]Recipe execution failed: {e}[/red]")
            raise typer.Exit(1)
    
    # Standard fine-tuning flow
    try:
        # Initialize trainer
        trainer = get_trainer(method, model_name, output_path)
        
        # Load model and tokenizer
        console.print("[cyan]Loading model and tokenizer...[/cyan]")
        trainer.load_model_and_tokenizer()
        
        # Apply LoRA/QLoRA
        console.print("[cyan]Applying LoRA adapters...[/cyan]")
        trainer.apply_lora()
        
        # Initialize dataset processor
        dataset_processor = get_dataset_processor(trainer.tokenizer)
        
        # Load dataset
        console.print(f"[cyan]Loading dataset: {dataset}[/cyan]")
        if dataset in ["gsm8k", "math", "mmlu"]:
            train_dataset = dataset_processor.load_math_dataset(dataset, max_samples=max_samples)
        else:
            train_dataset = dataset_processor.load_dataset(dataset, max_samples=max_samples)
        
        # Process dataset
        console.print("[cyan]Processing dataset...[/cyan]")
        processed_dataset = dataset_processor.process_dataset(train_dataset)
        
        # Update training settings if provided
        if epochs != settings.finetuning_num_epochs:
            settings.finetuning_num_epochs = epochs
        if batch_size != settings.finetuning_batch_size:
            settings.finetuning_batch_size = batch_size
        if learning_rate != settings.finetuning_learning_rate:
            settings.finetuning_learning_rate = learning_rate
        
        # Prepare trainer
        console.print("[cyan]Preparing trainer...[/cyan]")
        trainer.prepare_trainer(train_dataset=processed_dataset)
        
        # Train
        console.print("[green]Starting fine-tuning...[/green]")
        trainer.train()
        
        # Save model
        console.print(f"[cyan]Saving adapter to {output_path}[/cyan]")
        trainer.save_model()
        
        console.print(f"[green]Fine-tuning completed successfully![/green]")
        console.print(f"[dim]Adapter saved to: {output_path}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Fine-tuning failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def adapters(
    action: str = typer.Argument("list", help="Action: list, load, merge, export"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a", help="Adapter name"),
    base_model: Optional[str] = typer.Option(None, "--model", "-m", help="Base model name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
    format: str = typer.Option("gguf", "--format", "-f", help="Export format"),
) -> None:
    """Manage LoRA adapters."""
    _print_banner()
    
    try:
        from finetuning.adapter_manager import get_adapter_manager
    except ImportError as e:
        console.print(f"[red]Adapter manager not available: {e}[/red]")
        console.print("[yellow]Install with: pip install peft[/yellow]")
        raise typer.Exit(1)
    
    model_name = base_model or settings.local_model_name
    
    try:
        manager = get_adapter_manager(model_name)
        
        if action == "list":
            console.print("[cyan]Available adapters:[/cyan]")
            adapters = manager.list_adapters()
            if adapters:
                for adapter_name in adapters:
                    console.print(f"  • {adapter_name}")
            else:
                console.print("[dim]No adapters found[/dim]")
        
        elif action == "load":
            if not adapter:
                console.print("[red]Adapter name required for load action[/red]")
                raise typer.Exit(1)
            
            console.print(f"[cyan]Loading adapter: {adapter}[/cyan]")
            manager.load_base_model()
            loaded_model = manager.load_adapter(adapter)
            console.print(f"[green]Adapter {adapter} loaded successfully[/green]")
        
        elif action == "merge":
            if not adapter:
                console.print("[red]Adapter name required for merge action[/red]")
                raise typer.Exit(1)
            
            console.print(f"[cyan]Merging adapter: {adapter}[/cyan]")
            manager.load_base_model()
            manager.load_adapter(adapter)
            output_path = output or (settings.multi_lora_adapter_dir / f"{adapter}_merged")
            manager.merge_and_save(adapter, output_path)
            console.print(f"[green]Merged model saved to {output_path}[/green]")
        
        elif action == "export":
            if not adapter:
                console.print("[red]Adapter name required for export action[/red]")
                raise typer.Exit(1)
            
            console.print(f"[cyan]Exporting adapter: {adapter} to {format}[/cyan]")
            manager.load_base_model()
            manager.load_adapter(adapter)
            output_path = output or (settings.multi_lora_adapter_dir / f"{adapter}_{format}")
            manager.export_adapter(adapter, format, output_path)
            console.print(f"[green]Adapter exported to {output_path}[/green]")
        
        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("[dim]Available actions: list, load, merge, export[/dim]")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]Adapter management failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def merge(
    method: str = typer.Option("ties", "--method", "-m", help="Merging method: ties, dare, task_arithmetic, slerp, git_rebasin, fisher, regmean, model_stock"),
    models: list[str] = typer.Option(..., "--models", help="List of model paths or names to merge"),
    weights: Optional[list[float]] = typer.Option(None, "--weights", "-w", help="Optional weights for each model"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for merged model"),
    operations: Optional[list[str]] = typer.Option(None, "--operations", help="Operations for task_arithmetic: add, subtract"),
    interpolation: Optional[float] = typer.Option(None, "--interpolation", "-t", help="SLERP interpolation parameter"),
    drop_rate: Optional[float] = typer.Option(None, "--drop-rate", help="DARE drop rate"),
    trim_threshold: Optional[float] = typer.Option(None, "--trim-threshold", help="TIES trim threshold"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Merge multiple models using advanced merging algorithms."""
    _print_banner()
    
    try:
        from merging import (
            ModelMerger,
            TIESMerger,
            DAREMerger,
            TaskArithmetic,
            SLERPMerger,
            GitReBasin,
            FisherMerger,
            RegMeanMerger,
            ModelStock,
        )
    except ImportError as e:
        console.print(f"[red]Merging dependencies not installed: {e}[/red]")
        console.print("[yellow]Install with: pip install scipy[/yellow]")
        raise typer.Exit(1)
    
    output_path = output or settings.merging_output_dir
    
    console.print(f"[green]Merging {len(models)} models using {method}[/green]")
    console.print(f"[dim]Output: {output_path}[/dim]")
    
    try:
        # Select appropriate merger
        mergers = {
            "ties": TIESMerger,
            "dare": DAREMerger,
            "task_arithmetic": TaskArithmetic,
            "slerp": SLERPMerger,
            "git_rebasin": GitReBasin,
            "fisher": FisherMerger,
            "regmean": RegMeanMerger,
            "model_stock": ModelStock,
            "average": ModelMerger,
        }
        
        merger_class = mergers.get(method.lower())
        if merger_class is None:
            console.print(f"[red]Unknown merging method: {method}[/red]")
            console.print(f"[dim]Available methods: {', '.join(mergers.keys())}[/dim]")
            raise typer.Exit(1)
        
        # Initialize merger
        merger = merger_class()
        
        # Load models
        console.print("[cyan]Loading models...[/cyan]")
        for model_path in models:
            model_name = Path(model_path).name
            merger.load_model(model_path, model_name)
        
        # Perform merge
        console.print(f"[cyan]Applying {method} merging...[/cyan]")
        
        merge_kwargs = {}
        
        # Method-specific parameters
        if method == "ties":
            if trim_threshold is not None:
                merge_kwargs["trim_threshold"] = trim_threshold
        elif method == "dare":
            if drop_rate is not None:
                merge_kwargs["drop_rate"] = drop_rate
            if seed is not None:
                merge_kwargs["seed"] = seed
        elif method == "task_arithmetic":
            if operations is not None:
                merge_kwargs["operations"] = operations
        elif method == "slerp":
            if interpolation is not None:
                merge_kwargs["interpolation"] = interpolation
        elif method == "average":
            if weights is not None:
                merge_kwargs["weights"] = weights
        
        if weights is not None and method != "task_arithmetic":
            merge_kwargs["weights"] = weights
        
        merged_model = merger.merge(models, **merge_kwargs)
        
        # Save merged model
        console.print(f"[cyan]Saving merged model to {output_path}[/cyan]")
        
        # Load tokenizer from first model
        tokenizer = None
        if settings.merging_save_tokenizer:
            try:
                tokenizer = merger.load_tokenizer(models[0], "merged")
            except Exception as e:
                console.print(f"[yellow]Could not load tokenizer: {e}[/yellow]")
        
        merger.save_merged_model(merged_model, output_path, tokenizer)
        
        console.print(f"[green]Model merging completed successfully![/green]")
        console.print(f"[dim]Merged model saved to: {output_path}[/dim]")
        
        # Cleanup
        merger.cleanup()
        
    except Exception as e:
        console.print(f"[red]Model merging failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def advanced_benchmark(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name for evaluation"),
    no_advanced: bool = typer.Option(False, "--no-advanced", help="Disable advanced reasoning"),
    num_questions: int = typer.Option(10, "--num-questions", "-n", help="Number of questions per category"),
    categories: Optional[list[str]] = typer.Option(None, "--categories", "-c", help="Categories to test"),
    split: str = typer.Option("test", "--split", help="Dataset split (test or validation)"),
    output_dir: str = typer.Option("enhanced_benchmark_results", "--output", "-o", help="Output directory"),
    offline: bool = typer.Option(False, "--offline", help="Run in offline mode with simulated responses"),
    cutting_edge: Optional[bool] = typer.Option(None, "--cutting-edge", help="Use cutting-edge techniques (default: from config)"),
    difficulty: str = typer.Option("mixed", "--difficulty", "-d", help="Question difficulty: easy, medium, hard, mixed"),
) -> None:
    """Run enhanced MMLU-Pro benchmark with advanced reasoning integration."""
    # Use plain print for Windows compatibility
    print(f"Running Enhanced MMLU-Pro Benchmark")
    print(f"Model: {model or settings.default_model}")

    # Use config defaults if not specified
    use_advanced = not no_advanced
    use_cutting = cutting_edge if cutting_edge is not None else settings.use_cutting_edge_techniques

    print(f"Advanced Reasoning: {use_advanced}")
    if use_cutting:
        print(f"Mode: CUTTING-EDGE techniques enabled")
    print(f"Offline Mode: {offline}")
    print(f"Difficulty: {difficulty}")

    try:
        # Import the enhanced benchmark
        import importlib.util
        spec = importlib.util.spec_from_file_location("mmlu_pro_enhanced",
                                                       Path(__file__).parent / "mmlu_pro_enhanced.py")
        enhanced_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enhanced_module)

        # Run benchmark
        benchmark = enhanced_module.EnhancedMMLUProBenchmark(
            model=model,
            use_advanced_reasoning=use_advanced,
            num_questions=num_questions,
            categories=categories,
            split=split,
            output_dir=output_dir,
            offline_mode=offline,
            use_cutting_edge=use_cutting,
            difficulty=difficulty
        )

        results = benchmark.run_benchmark()

        print(f"Benchmark completed!")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"Benchmark failed: {e}")
        print(f"Make sure mmlu_pro_enhanced.py exists in the project directory")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default to `ui` if no sub-command given
    if len(sys.argv) == 1:
        sys.argv.append("ui")
    app()
