#!/usr/bin/env python3
"""
Code quality validation script for photonic attention system.

Validates code quality, structure, and best practices.
"""

import os
import sys
import ast
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path

class CodeQualityChecker:
    """Comprehensive code quality checker."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        
    def check_directory(self, directory: str) -> Dict[str, Any]:
        """Check entire directory for code quality issues."""
        print(f"📝 Checking code quality in: {directory}")
        
        results = {
            'files_checked': 0,
            'syntax_errors': 0,
            'quality_issues': 0,
            'warnings': 0,
            'lines_of_code': 0,
            'details': {
                'syntax_errors': [],
                'quality_issues': [],
                'warnings': []
            }
        }
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_result = self.check_file(file_path)
                    
                    results['files_checked'] += 1
                    results['lines_of_code'] += file_result['lines_of_code']
                    
                    if file_result['syntax_error']:
                        results['syntax_errors'] += 1
                        results['details']['syntax_errors'].append(
                            f"{file_path}: {file_result['syntax_error']}"
                        )
                    
                    results['quality_issues'] += len(file_result['quality_issues'])
                    results['warnings'] += len(file_result['warnings'])
                    
                    results['details']['quality_issues'].extend([
                        f"{file_path}: {issue}" for issue in file_result['quality_issues']
                    ])
                    results['details']['warnings'].extend([
                        f"{file_path}: {warning}" for warning in file_result['warnings']
                    ])
        
        return results
    
    def check_file(self, file_path: str) -> Dict[str, Any]:
        """Check individual file for quality issues."""
        result = {
            'lines_of_code': 0,
            'syntax_error': None,
            'quality_issues': [],
            'warnings': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count lines of code (non-empty, non-comment lines)
            lines = content.split('\n')
            result['lines_of_code'] = len([
                line for line in lines 
                if line.strip() and not line.strip().startswith('#')
            ])
            
            # Check syntax
            try:
                tree = ast.parse(content)
                self.check_ast_quality(tree, file_path, result)
            except SyntaxError as e:
                result['syntax_error'] = str(e)
                return result
            
            # Text-based quality checks
            self.check_text_quality(content, file_path, result)
            
        except Exception as e:
            result['warnings'].append(f"Could not analyze file: {e}")
        
        return result
    
    def check_ast_quality(self, tree: ast.AST, file_path: str, result: Dict[str, Any]) -> None:
        """Check AST for quality issues."""
        class_count = 0
        function_count = 0
        complexity_issues = []
        
        for node in ast.walk(tree):
            # Count classes and functions
            if isinstance(node, ast.ClassDef):
                class_count += 1
                if len(node.name) < 3:
                    result['warnings'].append(f"Short class name '{node.name}' at line {node.lineno}")
                    
            elif isinstance(node, ast.FunctionDef):
                function_count += 1
                
                # Check function complexity
                if self.calculate_complexity(node) > 10:
                    complexity_issues.append(f"High complexity function '{node.name}' at line {node.lineno}")
                
                # Check function length
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 50:
                    result['warnings'].append(f"Long function '{node.name}' ({func_lines} lines) at line {node.lineno}")
                
                # Check docstring
                if not ast.get_docstring(node):
                    result['warnings'].append(f"Missing docstring for function '{node.name}' at line {node.lineno}")
            
            elif isinstance(node, ast.ClassDef):
                # Check class docstring
                if not ast.get_docstring(node):
                    result['warnings'].append(f"Missing docstring for class '{node.name}' at line {node.lineno}")
        
        # Add complexity issues
        result['quality_issues'].extend(complexity_issues)
        
        # Check file structure
        if class_count == 0 and function_count == 0:
            result['warnings'].append("File contains no classes or functions")
        elif class_count > 10:
            result['warnings'].append(f"Too many classes ({class_count}) in single file")
        elif function_count > 20:
            result['warnings'].append(f"Too many functions ({function_count}) in single file")
    
    def calculate_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            # Add complexity for control flow
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def check_text_quality(self, content: str, file_path: str, result: Dict[str, Any]) -> None:
        """Check text-based quality issues."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 120:
                result['warnings'].append(f"Long line ({len(line)} chars) at line {i}")
            
            # Check for TODO comments
            if 'TODO' in line.upper():
                result['info'] = result.get('info', [])
                result['info'].append(f"TODO comment at line {i}")
            
            # Check for print statements (should use logging)
            if re.search(r'\bprint\s*\(', line) and 'test' not in file_path.lower():
                result['warnings'].append(f"Print statement found at line {i} (consider using logging)")
        
        # Check imports
        import_lines = [line for line in lines if line.strip().startswith(('import ', 'from '))]
        if len(import_lines) > 20:
            result['warnings'].append(f"Too many imports ({len(import_lines)})")
        
        # Check for duplicate imports
        import_set = set()
        for line in import_lines:
            if line in import_set:
                result['warnings'].append(f"Duplicate import: {line.strip()}")
            import_set.add(line)


def check_file_structure(src_dir: str) -> Dict[str, Any]:
    """Check overall file structure and organization."""
    print("📁 Checking file structure...")
    
    results = {
        'total_files': 0,
        'python_files': 0,
        'test_files': 0,
        'config_files': 0,
        'missing_init_files': [],
        'large_files': [],
        'empty_files': []
    }
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            
            results['total_files'] += 1
            
            if file.endswith('.py'):
                results['python_files'] += 1
                
                if 'test' in file.lower():
                    results['test_files'] += 1
                
                # Check for large files
                if file_size > 50 * 1024:  # 50KB
                    results['large_files'].append(f"{file_path} ({file_size // 1024}KB)")
                
                # Check for empty files
                if file_size == 0:
                    results['empty_files'].append(file_path)
            
            elif file.endswith(('.json', '.yaml', '.yml', '.toml')):
                results['config_files'] += 1
        
        # Check for missing __init__.py files
        if any(f.endswith('.py') for f in files) and '__init__.py' not in files:
            if 'test' not in root.lower():  # Test directories don't need __init__.py
                results['missing_init_files'].append(root)
    
    return results


def check_documentation_quality(src_dir: str) -> Dict[str, Any]:
    """Check documentation quality."""
    print("📚 Checking documentation quality...")
    
    results = {
        'files_with_docstrings': 0,
        'files_without_docstrings': 0,
        'functions_with_docstrings': 0,
        'functions_without_docstrings': 0,
        'classes_with_docstrings': 0,
        'classes_without_docstrings': 0
    }
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    # Check module docstring
                    if ast.get_docstring(tree):
                        results['files_with_docstrings'] += 1
                    else:
                        results['files_without_docstrings'] += 1
                    
                    # Check function and class docstrings
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if ast.get_docstring(node):
                                results['functions_with_docstrings'] += 1
                            else:
                                results['functions_without_docstrings'] += 1
                        
                        elif isinstance(node, ast.ClassDef):
                            if ast.get_docstring(node):
                                results['classes_with_docstrings'] += 1
                            else:
                                results['classes_without_docstrings'] += 1
                
                except Exception:
                    continue  # Skip files with syntax errors
    
    return results


def run_code_quality_check():
    """Run comprehensive code quality check."""
    print("📋 PHOTONIC FLASH ATTENTION - CODE QUALITY CHECK")
    print("=" * 65)
    
    checker = CodeQualityChecker()
    src_dir = 'src'
    
    # Main code quality check
    quality_results = checker.check_directory(src_dir)
    
    # File structure check
    structure_results = check_file_structure(src_dir)
    
    # Documentation check
    doc_results = check_documentation_quality(src_dir)
    
    # Print results
    print(f"\n📊 CODE QUALITY SUMMARY:")
    print(f"Files checked: {quality_results['files_checked']}")
    print(f"Lines of code: {quality_results['lines_of_code']:,}")
    print(f"Syntax errors: {quality_results['syntax_errors']}")
    print(f"Quality issues: {quality_results['quality_issues']}")
    print(f"Warnings: {quality_results['warnings']}")
    
    print(f"\n📁 FILE STRUCTURE:")
    print(f"Python files: {structure_results['python_files']}")
    print(f"Test files: {structure_results['test_files']}")
    print(f"Config files: {structure_results['config_files']}")
    print(f"Missing __init__.py: {len(structure_results['missing_init_files'])}")
    print(f"Large files (>50KB): {len(structure_results['large_files'])}")
    
    print(f"\n📚 DOCUMENTATION:")
    print(f"Files with docstrings: {doc_results['files_with_docstrings']}")
    print(f"Functions with docstrings: {doc_results['functions_with_docstrings']}")
    print(f"Classes with docstrings: {doc_results['classes_with_docstrings']}")
    
    # Calculate quality score
    quality_score = 100
    
    # Deduct for syntax errors
    if quality_results['syntax_errors'] > 0:
        quality_score -= quality_results['syntax_errors'] * 20
        print(f"\n❌ Syntax errors: -{quality_results['syntax_errors'] * 20} points")
    
    # Deduct for quality issues
    if quality_results['quality_issues'] > 10:
        quality_score -= min((quality_results['quality_issues'] - 10) * 2, 20)
        print(f"⚠️ Quality issues: -{min((quality_results['quality_issues'] - 10) * 2, 20)} points")
    
    # Deduct for too many warnings
    if quality_results['warnings'] > 50:
        quality_score -= min((quality_results['warnings'] - 50), 20)
        print(f"⚠️ Too many warnings: -{min((quality_results['warnings'] - 50), 20)} points")
    
    # Deduct for missing documentation
    total_functions = doc_results['functions_with_docstrings'] + doc_results['functions_without_docstrings']
    total_classes = doc_results['classes_with_docstrings'] + doc_results['classes_without_docstrings']
    
    if total_functions > 0:
        doc_coverage = doc_results['functions_with_docstrings'] / total_functions
        if doc_coverage < 0.7:
            quality_score -= int((0.7 - doc_coverage) * 30)
            print(f"⚠️ Low documentation coverage: -{int((0.7 - doc_coverage) * 30)} points")
    
    # Bonus for good structure
    if len(structure_results['missing_init_files']) == 0:
        print("✅ All packages have __init__.py files")
    else:
        quality_score -= len(structure_results['missing_init_files']) * 2
        print(f"⚠️ Missing __init__.py files: -{len(structure_results['missing_init_files']) * 2} points")
    
    quality_score = max(0, quality_score)
    
    print(f"\n🏆 CODE QUALITY SCORE: {quality_score}/100")
    
    if quality_score >= 85:
        print("🟢 EXCELLENT - High quality codebase")
        return True
    elif quality_score >= 70:
        print("🟡 GOOD - Minor improvements recommended")
        return True
    elif quality_score >= 50:
        print("🟠 FAIR - Code quality improvements needed")
        return False
    else:
        print("🔴 POOR - Significant code quality issues")
        return False


if __name__ == "__main__":
    success = run_code_quality_check()
    sys.exit(0 if success else 1)