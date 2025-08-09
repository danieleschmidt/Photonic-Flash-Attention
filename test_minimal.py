#!/usr/bin/env python3
"""
Minimal test without external dependencies to verify code structure.
"""

import os
import sys
import importlib.util

def test_code_structure():
    """Test that all Python files can be parsed and have valid syntax."""
    print("🧪 Testing code structure and syntax...")
    
    errors = []
    file_count = 0
    
    # Walk through src directory
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_count += 1
                
                try:
                    # Test syntax by compiling
                    with open(file_path, 'r') as f:
                        source_code = f.read()
                    
                    compile(source_code, file_path, 'exec')
                    print(f"   ✅ {os.path.relpath(file_path)}")
                    
                except SyntaxError as e:
                    error_msg = f"Syntax error in {file_path}:{e.lineno}: {e.msg}"
                    errors.append(error_msg)
                    print(f"   ❌ {os.path.relpath(file_path)}: {e.msg}")
                    
                except Exception as e:
                    error_msg = f"Error in {file_path}: {e}"
                    errors.append(error_msg)
                    print(f"   ❌ {os.path.relpath(file_path)}: {e}")
    
    print(f"\nTested {file_count} Python files")
    
    if errors:
        print(f"❌ {len(errors)} error(s) found:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True


def test_import_structure():
    """Test import structure without executing code."""
    print("\n🧪 Testing import structure...")
    
    # Key files that should exist
    key_files = [
        'src/photonic_flash_attention/__init__.py',
        'src/photonic_flash_attention/config.py',
        'src/photonic_flash_attention/cli.py',
        'src/photonic_flash_attention/core/flash_attention_3.py',
        'src/photonic_flash_attention/core/photonic_attention.py',
        'src/photonic_flash_attention/core/hybrid_router.py',
        'src/photonic_flash_attention/integration/pytorch/modules.py',
        'src/photonic_flash_attention/photonic/hardware/detection.py',
        'src/photonic_flash_attention/utils/logging.py',
        'src/photonic_flash_attention/utils/exceptions.py',
        'src/photonic_flash_attention/utils/validation.py',
    ]
    
    missing_files = []
    
    for file_path in key_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"   ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"   ❌ {file_path} (missing)")
    
    if missing_files:
        print(f"❌ {len(missing_files)} key file(s) missing")
        return False
    else:
        print("✅ All key files present")
        return True


def test_configuration_files():
    """Test configuration files."""
    print("\n🧪 Testing configuration files...")
    
    config_files = [
        'pyproject.toml',
        'setup.py', 
        'requirements.txt',
        'README.md',
        'LICENSE',
    ]
    
    missing_configs = []
    
    for config_file in config_files:
        full_path = os.path.join(os.path.dirname(__file__), config_file)
        if os.path.exists(full_path):
            print(f"   ✅ {config_file}")
        else:
            missing_configs.append(config_file)
            print(f"   ❌ {config_file} (missing)")
    
    if missing_configs:
        print(f"❌ {len(missing_configs)} config file(s) missing")
        return False
    else:
        print("✅ All config files present")
        return True


def test_example_files():
    """Test example files."""
    print("\n🧪 Testing example files...")
    
    example_files = [
        'examples/basic_usage.py',
        'examples/transformer_integration.py',
    ]
    
    missing_examples = []
    
    for example_file in example_files:
        full_path = os.path.join(os.path.dirname(__file__), example_file)
        if os.path.exists(full_path):
            # Test syntax
            try:
                with open(full_path, 'r') as f:
                    source_code = f.read()
                compile(source_code, full_path, 'exec')
                print(f"   ✅ {example_file}")
            except SyntaxError as e:
                print(f"   ❌ {example_file}: syntax error at line {e.lineno}")
        else:
            missing_examples.append(example_file)
            print(f"   ❌ {example_file} (missing)")
    
    if missing_examples:
        print(f"❌ {len(missing_examples)} example file(s) missing")
        return False
    else:
        print("✅ All example files present and valid")
        return True


def test_documentation():
    """Test documentation structure."""
    print("\n🧪 Testing documentation...")
    
    doc_files = [
        'docs/index.rst',
        'docs/conf.py',
        'docs/quickstart.rst',
    ]
    
    docs_valid = True
    
    for doc_file in doc_files:
        full_path = os.path.join(os.path.dirname(__file__), doc_file)
        if os.path.exists(full_path):
            print(f"   ✅ {doc_file}")
        else:
            print(f"   ⚠️ {doc_file} (missing but optional)")
    
    # Check README
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        # Check for key sections
        required_sections = [
            '# Photonic-Flash-Attention',
            '## Quick Start',
            '## Installation',
            '## Usage',
        ]
        
        for section in required_sections:
            if section in readme_content:
                print(f"   ✅ README section: {section}")
            else:
                print(f"   ⚠️ README section missing: {section}")
                docs_valid = False
    
    print("✅ Documentation structure checked")
    return docs_valid


def test_package_structure():
    """Test overall package structure."""
    print("\n🧪 Testing package structure...")
    
    # Check directory structure
    expected_dirs = [
        'src/photonic_flash_attention',
        'src/photonic_flash_attention/core',
        'src/photonic_flash_attention/photonic',
        'src/photonic_flash_attention/photonic/hardware',
        'src/photonic_flash_attention/photonic/optical_kernels',
        'src/photonic_flash_attention/integration',
        'src/photonic_flash_attention/integration/pytorch',
        'src/photonic_flash_attention/utils',
        'examples',
        'tests',
        'docs',
    ]
    
    structure_valid = True
    
    for dir_path in expected_dirs:
        full_path = os.path.join(os.path.dirname(__file__), dir_path)
        if os.path.exists(full_path):
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ (missing)")
            structure_valid = False
    
    if structure_valid:
        print("✅ Package structure valid")
    else:
        print("❌ Package structure has issues")
    
    return structure_valid


def main():
    """Run minimal tests."""
    print("🚀 Photonic Flash Attention - Minimal Test Suite")
    print("="*60)
    print("(Testing without external dependencies)")
    print()
    
    tests = [
        ("Code Structure", test_code_structure),
        ("Import Structure", test_import_structure), 
        ("Configuration Files", test_configuration_files),
        ("Example Files", test_example_files),
        ("Documentation", test_documentation),
        ("Package Structure", test_package_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 MINIMAL TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print("-"*60)
    print(f"Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 ALL MINIMAL TESTS PASSED!")
        print("   Code structure and syntax are valid.")
        print("   Package is properly structured.")
        return 0
    else:
        print(f"\n⚠️ {total-passed} test(s) failed.")
        print("   Fix structural issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())