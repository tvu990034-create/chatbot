#!/usr/bin/env python
"""
Deployment Automation Script for Local Chatbot API
Automates deployment tasks including environment setup, validation, and monitoring
"""

import os
import sys
import subprocess
import json
import time
import requests
from pathlib import Path

class DeploymentManager:
    """Automated deployment management."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_file = self.base_dir / "config.json"
        self.venv_dir = self.base_dir / ".venv"
        self.logs_dir = self.base_dir / "logs"
        
    def check_environment(self) -> bool:
        """Check deployment environment prerequisites."""
        print("=== Environment Check ===")
        
        checks = []
        
        # Python version check
        python_version = sys.version_info
        python_ok = python_version >= (3, 8)
        checks.append(("Python Version", python_ok, f"{'.'.join(map(str, python_version[:3]))}"))
        
        # Virtual environment check
        venv_exists = self.venv_dir.exists()
        checks.append(("Virtual Environment", venv_exists, str(self.venv_dir)))
        
        # Ollama check
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            ollama_ok = response.status_code == 200
            checks.append(("Ollama Service", ollama_ok, "Running"))
        except:
            ollama_ok = False
            checks.append(("Ollama Service", ollama_ok, "Not running"))
        
        # Configuration file check
        config_exists = self.config_file.exists()
        checks.append(("Configuration File", config_exists, str(self.config_file)))
        
        # Logs directory
        self.logs_dir.mkdir(exist_ok=True)
        checks.append(("Logs Directory", True, str(self.logs_dir)))
        
        # Print results
        for check_name, passed, detail in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}: {detail}")
        
        all_passed = all(check[1] for check in checks)
        print(f"\nEnvironment Check: {'PASSED' if all_passed else 'FAILED'}")
        return all_passed
    
    def setup_environment(self):
        """Setup deployment environment."""
        print("\n=== Environment Setup ===")
        
        # Create virtual environment if needed
        if not self.venv_dir.exists():
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(self.venv_dir)], check=True)
            print("Virtual environment created")
        
        # Install dependencies
        print("Installing dependencies...")
        pip_path = self.venv_dir / "Scripts" / "pip.exe" if os.name == 'nt' else self.venv_dir / "bin" / "pip"
        
        try:
            subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
            print("Dependencies installed")
        except:
            # Install manually if requirements.txt not found
            packages = ["fastapi", "uvicorn", "requests", "pydantic", "litellm"]
            subprocess.run([str(pip_path), "install"] + packages, check=True)
            print("Core dependencies installed")
        
        # Generate default configuration
        if not self.config_file.exists():
            print("Generating default configuration...")
            subprocess.run([str(pip_path.parent / "python.exe"), "config_manager.py"], check=True)
            print("Configuration generated")
    
    def validate_deployment(self) -> bool:
        """Validate deployment by running tests."""
        print("\n=== Deployment Validation ===")
        
        try:
            # Run system tests
            test_result = subprocess.run(
                [str(self.venv_dir / "Scripts" / "python.exe"), "test_system.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            print(test_result.stdout)
            
            if test_result.returncode == 0:
                print("✅ Deployment validation PASSED")
                return True
            else:
                print("❌ Deployment validation FAILED")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Deployment validation TIMEOUT")
            return False
        except Exception as e:
            print(f"❌ Deployment validation ERROR: {e}")
            return False
    
    def start_service(self):
        """Start the API service."""
        print("\n=== Starting Service ===")
        
        python_path = self.venv_dir / "Scripts" / "python.exe" if os.name == 'nt' else self.venv_dir / "bin" / "python"
        
        # Start service in background
        process = subprocess.Popen(
            [str(python_path), "api_service/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"Service started with PID: {process.pid}")
        
        # Wait for service to be ready
        print("Waiting for service to be ready...")
        time.sleep(5)
        
        # Check if service is running
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
            if response.status_code == 200:
                print("✅ Service is running and healthy")
                return process
            else:
                print("❌ Service returned unhealthy status")
                process.terminate()
                return None
        except:
            print("❌ Service failed to start")
            process.terminate()
            return None
    
    def stop_service(self, process):
        """Stop the API service."""
        print("\n=== Stopping Service ===")
        
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ Service stopped gracefully")
            except:
                process.kill()
                print("⚠️ Service stopped forcefully")
    
    def health_check(self):
        """Perform health check on running service."""
        print("\n=== Health Check ===")
        
        try:
            response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
            health_data = response.json()
            
            print(f"Status: {health_data.get('status')}")
            print(f"Available Models: {health_data.get('available_models')}")
            print(f"Optimizations: {health_data.get('optimizations_active')}")
            
            return health_data.get('status') == "healthy"
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def create_backup(self):
        """Create backup of current configuration and data."""
        print("\n=== Creating Backup ===")
        
        backup_dir = self.base_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"backup_{timestamp}.tar.gz"
        
        # Create backup (simplified - just config)
        import shutil
        if self.config_file.exists():
            shutil.copy2(self.config_file, backup_dir / f"config_{timestamp}.json")
            print(f"✅ Configuration backed up to {backup_dir / f'config_{timestamp}.json'}")
        
        return backup_dir
    
    def get_deployment_info(self):
        """Get deployment information."""
        print("\n=== Deployment Information ===")
        
        info = {
            "deployment_dir": str(self.base_dir),
            "config_file": str(self.config_file),
            "venv_dir": str(self.venv_dir),
            "logs_dir": str(self.logs_dir),
            "python_version": sys.version,
            "platform": sys.platform,
            "config": {}
        }
        
        if self.config_file.exists():
            with open(self.config_file) as f:
                info["config"] = json.load(f)
        
        print(f"Deployment Directory: {info['deployment_dir']}")
        print(f"Python Version: {info['python_version']}")
        print(f"Platform: {info['platform']}")
        print(f"Configuration: {info['config_file']}")
        
        return info

def main():
    """Main deployment automation."""
    print("=" * 60)
    print("AUTOMATED DEPLOYMENT MANAGER")
    print("=" * 60)
    
    manager = DeploymentManager()
    
    # Step 1: Environment Check
    if not manager.check_environment():
        print("❌ Environment check failed. Fix issues before proceeding.")
        return False
    
    # Step 2: Environment Setup
    print("\nProceeding with environment setup...")
    manager.setup_environment()
    
    # Step 3: Deployment Validation
    if not manager.validate_deployment():
        print("❌ Deployment validation failed.")
        return False
    
    # Step 4: Start Service
    service_process = manager.start_service()
    if not service_process:
        print("❌ Failed to start service.")
        return False
    
    # Step 5: Health Check
    if not manager.health_check():
        print("❌ Health check failed.")
        manager.stop_service(service_process)
        return False
    
    # Step 6: Backup
    manager.create_backup()
    
    # Step 7: Deployment Info
    manager.get_deployment_info()
    
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUCCESSFUL")
    print("=" * 60)
    print("\nService is running at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Dashboard: http://localhost:8000/api/v1/dashboard")
    print("\nPress Ctrl+C to stop the service")
    
    try:
        # Keep service running
        service_process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down service...")
        manager.stop_service(service_process)
        print("Deployment complete.")

if __name__ == "__main__":
    main()