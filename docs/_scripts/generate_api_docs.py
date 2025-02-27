#!/usr/bin/env python3
"""
Script to generate API documentation for NeuroBayes using pydoc-markdown
"""
import os
import sys
import subprocess

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Configuration
output_dir = os.path.join(os.path.dirname(__file__), "..", "api")
jekyll_template = """---
layout: default
title: {title}
parent: {parent}
grand_parent: API Reference
---

{content}
"""

# Create directories
categories = ["models", "networks", "utils"]
for category in categories:
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

# Create main index file
with open(os.path.join(output_dir, "index.md"), "w") as f:
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

# Create category index files
for category in categories:
    with open(os.path.join(output_dir, category, "index.md"), "w") as f:
        f.write(f"""---
layout: default
title: {category.capitalize()}
parent: API Reference
has_children: true
---

# {category.capitalize()} API Reference

This section documents the {category} modules in NeuroBayes.
""")

# Define modules to document with their categories and titles
modules_to_document = [
    # Models
    ("neurobayes.models.bnn", "models", "BNN"),
    ("neurobayes.models.partial_bnn", "models", "PartialBNN"),
    ("neurobayes.models.gp", "models", "GP"),
    ("neurobayes.models.dkl", "models", "DKL"),
    ("neurobayes.models.vigp", "models", "VIGP"),
    ("neurobayes.models.vidkl", "models", "VIDKL"),
    ("neurobayes.models.bnn_heteroskedastic", "models", "HeteroskedasticBNN"),
    ("neurobayes.models.bnn_heteroskedastic_model", "models", "VarianceModelHeteroskedasticBNN"),
    ("neurobayes.models.kernels", "models", "Kernels"),
    
    # Networks
    ("neurobayes.flax_nets.mlp", "networks", "FlaxMLP"),
    ("neurobayes.flax_nets.convnet", "networks", "FlaxConvNet"),
    ("neurobayes.flax_nets.transformer", "networks", "FlaxTransformer"),
    ("neurobayes.flax_nets.deterministic_nn", "networks", "DeterministicNN"),
    
    # Utils
    ("neurobayes.utils.utils", "utils", "Utils"),
    ("neurobayes.utils.priors", "utils", "Priors"),
    ("neurobayes.utils.diagnostics", "utils", "Diagnostics")
]

# Process each module
for module_name, category, title in modules_to_document:
    print(f"Processing {module_name} -> {title}")
    
    # Run pydoc-markdown with the module name directly
    result = subprocess.run(
        ["pydoc-markdown", "-m", module_name],
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.returncode == 0 and result.stdout.strip():
        # Write the output to a markdown file
        module_short_name = module_name.split('.')[-1]
        output_file = os.path.join(output_dir, category, f"{module_short_name}.md")
        
        with open(output_file, "w") as f:
            f.write(jekyll_template.format(
                title=title,
                parent=category.capitalize(),
                content=result.stdout
            ))
        print(f"Successfully generated docs for {module_name}")
    else:
        print(f"Error or empty output for {module_name}:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

print("API documentation generation complete!")