"""
Dependency Checker

Checks and reports on installed Python packages and their versions:
- Module availability check
- Version reporting
- Missing dependency detection
- Package listing

Used to verify the environment has all required dependencies
for the Methodik v3 pipeline.
"""

import importlib
import pkg_resources
import sys
import subprocess

def check_module(module_name):
    """
    Check if a module is installed and get its version
    """
    try:
        module = importlib.import_module(module_name)
        try:
            version = pkg_resources.get_distribution(module_name).version
        except pkg_resources.DistributionNotFound:
            version = getattr(module, '__version__', 'Unknown')
        return True, version
    except ImportError:
        return False, None

def get_all_installed_packages():
    """
    Get all installed packages
    """
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def check_astronomical_dependencies():
    """
    Check astronomical and image processing dependencies
    """
    dependencies = [
        # Core Scientific Computing
        'numpy', 'scipy', 'pandas',
        
        # Astronomical Libraries
        'astropy', 'photutils',
        
        # Image Processing
        'scikit-image', 'opencv-python', 'imageio', 'pillow', 'matplotlib',
        
        # Machine Learning
        'scikit-learn',
        
        # Visualization
        'seaborn',
        
        # Configuration
        'pyyaml', 'jsonschema',
        
        # Wavelet Transform
        'PyWavelets',
        
        # Development & Testing
        'pytest', 'pytest-cov',
        
        # Performance
        'memory-profiler',
        
        # Optional: Parallel Processing
        'dask'
    ]
    
    print("Dependency Check Results:")
    print("-" * 50)
    
    missing_deps = []
    for dep in dependencies:
        installed, version = check_module(dep)
        status = "✓" if installed else "✗"
        print(f"{status} {dep}: {version if installed else 'Not Installed'}")
        
        if not installed:
            missing_deps.append(dep)
    
    print("\nMissing Dependencies:")
    print("-" * 50)
    for dep in missing_deps:
        print(dep)
    
    return missing_deps

def generate_requirements():
    """
    Generate an updated requirements.txt
    """
    installed_packages = get_all_installed_packages()
    
    # Core dependencies with specific versions
    core_dependencies = [
        ('numpy', '>=2.0.0'),
        ('scipy', '>=1.10.0'),
        ('pandas', '>=2.0.0'),
        ('astropy', '>=5.2.0'),
        ('scikit-image', '>=0.21.0'),
        ('scikit-learn', '>=1.2.0'),
        ('matplotlib', '>=3.7.0'),
        ('imageio', '[all]>=2.31.0'),
        ('pillow', '>=9.5.0'),
        ('pytest', '>=7.3.0'),
    ]
    
    requirements_content = "# Automatically generated requirements\n\n"
    
    for package, version_constraint in core_dependencies:
        if package in installed_packages:
            current_version = installed_packages[package]
            requirements_content += f"{package}{version_constraint}\n"
    
    # Write to requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("\nUpdated requirements.txt generated.")

def main():
    print("Python Version:", sys.version)
    
    # Check dependencies
    missing_deps = check_astronomical_dependencies()
    
    # Generate updated requirements
    generate_requirements()
    
    if missing_deps:
        print("\nTo install missing dependencies, run:")
        print("pip install " + " ".join(missing_deps))
        sys.exit(1)

if __name__ == '__main__':
    main()