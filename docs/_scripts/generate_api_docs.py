#!/usr/bin/env python3
"""
Script to generate API documentation for NeuroBayes
"""
import os
import subprocess
import importlib
import pkgutil
import inspect
import neurobayes  # Import your package

# Configuration
output_dir = "docs/api"
jekyll_template = """---
layout: default
title: {title}
parent: {parent}
grand_parent: API Reference
---

{content}
"""

# Category mapping based on your structure
category_mapping = {
    "models": ["neurobayes.models"],
    "networks": ["neurobayes.flax_nets"],
    "utils": ["neurobayes.utils"]
}

# Create directories if they don't exist
categories = ["models", "networks", "utils"]
for category in categories:
    os.makedirs(f"{output_dir}/{category}", exist_ok=True)

# Create an index file for API reference
with open(f"{output_dir}/index.md", "w") as f:
    f.write("""---
layout: default
title: API Reference
nav_order: 7
has_children: true
---

# API Reference

This section provides detailed API documentation for NeuroBayes classes and functions, automatically generated from the source code.

## Module Structure

- [Models](models/) - Bayesian and deterministic model implementations
- [Networks](networks/) - Neural network architectures 
- [Utils](utils/) - Utility functions and helpers

For conceptual explanations and usage examples, please refer to the corresponding sections in the main documentation.
""")

# Create index files for each category
for category in categories:
    with open(f"{output_dir}/{category}/index.md", "w") as f:
        f.write(f"""---
layout: default
title: {category.capitalize()}
parent: API Reference
has_children: true
---

# {category.capitalize()} API Reference

This section documents the {category} modules in NeuroBayes.
""")

# Function to get module category
def get_module_category(module_name):
    for category, prefixes in category_mapping.items():
        if any(module_name.startswith(prefix) for prefix in prefixes):
            return category
    return None

# Function to get main class from module
def get_main_class(module_name):
    try:
        module = importlib.import_module(module_name)
        # Look for classes in the module
        classes = [name for name, obj in inspect.getmembers(module, inspect.isclass)
                  if obj.__module__ == module_name]
        
        if classes:
            # Try to find a class with similar name as the module
            module_short_name = module_name.split('.')[-1]
            for class_name in classes:
                if class_name.lower() == module_short_name.lower() or \
                   module_short_name.lower() in class_name.lower():
                    return class_name
            # Otherwise return the first class
            return classes[0]
        else:
            # Fallback to module name
            return module_name.split('.')[-1].title().replace('_', '')
    except Exception as e:
        print(f"Error inspecting module {module_name}: {e}")
        return module_name.split('.')[-1].title().replace('_', '')

# Process modules by walking through the package
def process_package(package_name, package_path):
    for _, name, ispkg in pkgutil.iter_modules(package_path):
        full_name = f"{package_name}.{name}"
        
        if ispkg:
            # Recursively process subpackages
            try:
                subpackage = importlib.import_module(full_name)
                process_package(full_name, subpackage.__path__)
            except ImportError as e:
                print(f"Error importing {full_name}: {e}")
        else:
            # Process module
            category = get_module_category(full_name)
            if not category:
                print(f"Skipping module {full_name} (no category)")
                continue
                
            title = get_main_class(full_name)
            print(f"Generating docs for {full_name} -> {title} ({category})")
            
            result = subprocess.run(
                ["pydoc-markdown", "-m", full_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                content = result.stdout
                if content.strip():
                    filename = name.lower() + ".md"
                    with open(f"{output_dir}/{category}/{filename}", "w") as f:
                        f.write(jekyll_template.format(
                            title=title,
                            parent=category.capitalize(),
                            content=content
                        ))
                else:
                    print(f"Empty content for {full_name}, skipping")
            else:
                print(f"Error generating docs for {full_name}: {result.stderr}")

# Start processing from the main package
process_package("neurobayes", neurobayes.__path__)

print("API documentation generation complete!")