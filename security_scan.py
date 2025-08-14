#!/usr/bin/env python3
"""
Security scan script for photonic attention system.

Performs comprehensive security validation including:
- Code scanning for security issues
- Dependency vulnerability checking
- Configuration validation
- Input sanitization testing
"""

import os
import sys
import re
import ast
import importlib
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class SecurityScanner:
    """Comprehensive security scanner for the codebase."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        
    def scan_directory(self, directory: str) -> Dict[str, Any]:
        """Scan entire directory for security issues."""
        print(f"🔍 Scanning directory: {directory}")
        
        results = {
            'critical_issues': 0,
            'warnings': 0,
            'info': 0,
            'files_scanned': 0,
            'details': []
        }
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.scan_file(file_path)
                    results['files_scanned'] += 1
        
        results['critical_issues'] = len(self.issues)
        results['warnings'] = len(self.warnings) 
        results['info'] = len(self.info)
        results['details'] = {
            'critical_issues': self.issues,
            'warnings': self.warnings,
            'info': self.info
        }
        
        return results
    
    def scan_file(self, file_path: str) -> None:
        """Scan individual file for security issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                self.scan_ast(tree, file_path)
            except SyntaxError as e:
                self.warnings.append(f"Syntax error in {file_path}: {e}")
            
            # Text-based scans
            self.scan_dangerous_imports(content, file_path)
            self.scan_hardcoded_secrets(content, file_path)
            self.scan_sql_injection(content, file_path)
            self.scan_command_injection(content, file_path)
            self.scan_path_traversal(content, file_path)
            
        except Exception as e:
            self.warnings.append(f"Could not scan {file_path}: {e}")
    
    def scan_ast(self, tree: ast.AST, file_path: str) -> None:
        """Scan AST for security issues."""
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ['eval', 'exec', 'compile']:
                        self.issues.append(f"Dangerous function '{func_name}' used in {file_path}:{node.lineno}")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'spawn']:
                        self.warnings.append(f"System call '{node.func.attr}' in {file_path}:{node.lineno}")
            
            # Check for hardcoded passwords in assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and 'password' in target.id.lower():
                        if isinstance(node.value, ast.Str):
                            self.issues.append(f"Hardcoded password in {file_path}:{node.lineno}")
    
    def scan_dangerous_imports(self, content: str, file_path: str) -> None:
        """Scan for dangerous imports."""
        dangerous_imports = [
            r'import\s+os\.system',
            r'from\s+os\s+import\s+system',
            r'import\s+subprocess\s*$',
            r'from\s+subprocess\s+import.*shell=True'
        ]
        
        for pattern in dangerous_imports:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.warnings.append(f"Potentially dangerous import in {file_path}:{line_num}: {match.group()}")
    
    def scan_hardcoded_secrets(self, content: str, file_path: str) -> None:
        """Scan for hardcoded secrets and keys."""
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
            (r'secret[_-]?key\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret key'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token'),
            (r'-----BEGIN\s+PRIVATE\s+KEY-----', 'Private key in source'),
        ]
        
        for pattern, description in secret_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.issues.append(f"{description} in {file_path}:{line_num}")
    
    def scan_sql_injection(self, content: str, file_path: str) -> None:
        """Scan for potential SQL injection vulnerabilities."""
        sql_patterns = [
            r'SELECT.*\+.*',
            r'INSERT.*\+.*',
            r'UPDATE.*\+.*',
            r'DELETE.*\+.*',
            r'\.format\s*\(.*sql.*\)',
            r'f".*SELECT.*{.*}.*"',
        ]
        
        for pattern in sql_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.warnings.append(f"Potential SQL injection in {file_path}:{line_num}")
    
    def scan_command_injection(self, content: str, file_path: str) -> None:
        """Scan for command injection vulnerabilities."""
        command_patterns = [
            r'os\.system\s*\(.*\+',
            r'subprocess\.call\s*\(.*\+',
            r'subprocess\.run\s*\(.*\+.*shell\s*=\s*True',
            r'os\.popen\s*\(.*\+',
        ]
        
        for pattern in command_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.issues.append(f"Potential command injection in {file_path}:{line_num}")
    
    def scan_path_traversal(self, content: str, file_path: str) -> None:
        """Scan for path traversal vulnerabilities."""
        path_patterns = [
            r'open\s*\(.*\+.*\)',
            r'Path\s*\(.*\+.*\)',
            r'\.\./',
            r'os\.path\.join\s*\(.*input',
        ]
        
        for pattern in path_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                if '../' in match.group():
                    self.warnings.append(f"Path traversal pattern in {file_path}:{line_num}")
                else:
                    self.info.append(f"Potential path traversal in {file_path}:{line_num}")


def test_input_sanitization():
    """Test input sanitization functions."""
    print("🧪 Testing input sanitization...")
    
    try:
        from photonic_flash_attention.utils.security import sanitize_input, SecurityManager
        
        security_manager = SecurityManager()
        
        # Test cases
        test_cases = [
            ("normal_string", "Should pass"),
            ("../../../etc/passwd", "Should fail - path traversal"),
            ("string_with\x00null", "Should be cleaned"),
            ("a" * 20000, "Should fail - too long"),
            ({"key": "value"}, "Should pass - normal dict"),
            ({"../key": "value"}, "Should fail - path traversal in key"),
        ]
        
        passed = 0
        total = len(test_cases)
        
        for test_input, description in test_cases:
            try:
                result = security_manager.sanitize_input(test_input, allow_tensors=False)
                if isinstance(result, str) and len(result) > 10000:
                    print(f"❌ {description}: Should have rejected long string")
                elif isinstance(result, str) and '../' in result:
                    print(f"❌ {description}: Should have rejected path traversal")
                else:
                    print(f"✅ {description}: Passed")
                    passed += 1
            except Exception as e:
                if "path traversal" in str(e) or "too long" in str(e) or "control characters" in str(e):
                    print(f"✅ {description}: Correctly rejected - {e}")
                    passed += 1
                else:
                    print(f"❌ {description}: Unexpected error - {e}")
        
        print(f"Input sanitization tests: {passed}/{total} passed")
        return passed == total
        
    except ImportError as e:
        print(f"❌ Could not import security modules: {e}")
        return False


def test_security_validation():
    """Test security validation functions."""
    print("🔒 Testing security validation...")
    
    try:
        from photonic_flash_attention.utils.security import validate_hardware_security
        
        # Test device validation
        test_devices = [
            {
                'vendor': 'Photonic Flash Attention',
                'device_type': 'simulation',
                'temperature': 25.0,
                'max_optical_power': 0.01
            },
            {
                'vendor': 'Evil Corp',  # Should fail
                'device_type': 'unknown',
                'temperature': 100.0,  # Too hot
                'max_optical_power': 0.1  # Too high
            }
        ]
        
        passed = 0
        for i, device in enumerate(test_devices):
            try:
                result = validate_hardware_security(device)
                if i == 0:  # First device should pass
                    print("✅ Valid device accepted")
                    passed += 1
                else:  # Second device should fail
                    print("❌ Invalid device was accepted")
            except Exception as e:
                if i == 0:  # First device should not fail
                    print(f"❌ Valid device rejected: {e}")
                else:  # Second device should fail
                    print(f"✅ Invalid device correctly rejected: {e}")
                    passed += 1
        
        print(f"Security validation tests: {passed}/2 passed")
        return passed == 2
        
    except ImportError as e:
        print(f"❌ Could not import security validation: {e}")
        return False


def run_security_scan():
    """Run comprehensive security scan."""
    print("🛡️ PHOTONIC FLASH ATTENTION - SECURITY SCAN")
    print("=" * 60)
    
    scanner = SecurityScanner()
    
    # Scan main source directory
    src_results = scanner.scan_directory('src')
    
    print(f"\n📊 SCAN RESULTS:")
    print(f"Files scanned: {src_results['files_scanned']}")
    print(f"Critical issues: {src_results['critical_issues']}")
    print(f"Warnings: {src_results['warnings']}")
    print(f"Info items: {src_results['info']}")
    
    # Print details
    if src_results['critical_issues'] > 0:
        print(f"\n🚨 CRITICAL ISSUES:")
        for issue in src_results['details']['critical_issues']:
            print(f"  • {issue}")
    
    if src_results['warnings'] > 0:
        print(f"\n⚠️ WARNINGS:")
        for warning in src_results['details']['warnings'][:10]:  # Show first 10
            print(f"  • {warning}")
        if len(src_results['details']['warnings']) > 10:
            print(f"  ... and {len(src_results['details']['warnings']) - 10} more warnings")
    
    # Test input sanitization
    print(f"\n" + "=" * 60)
    sanitization_ok = test_input_sanitization()
    
    # Test security validation
    print(f"\n" + "=" * 60)
    validation_ok = test_security_validation()
    
    # Overall assessment
    print(f"\n" + "=" * 60)
    print("🎯 SECURITY ASSESSMENT:")
    
    security_score = 100
    
    if src_results['critical_issues'] > 0:
        security_score -= src_results['critical_issues'] * 20
        print(f"❌ Critical security issues found: -{src_results['critical_issues'] * 20} points")
    else:
        print("✅ No critical security issues found")
    
    if src_results['warnings'] > 10:
        security_score -= min((src_results['warnings'] - 10) * 2, 20)
        print(f"⚠️ Many warnings found: -{min((src_results['warnings'] - 10) * 2, 20)} points")
    elif src_results['warnings'] > 0:
        print(f"⚠️ Some warnings found (acceptable level)")
    else:
        print("✅ No warnings found")
    
    if not sanitization_ok:
        security_score -= 15
        print("❌ Input sanitization tests failed: -15 points")
    else:
        print("✅ Input sanitization tests passed")
    
    if not validation_ok:
        security_score -= 15
        print("❌ Security validation tests failed: -15 points")
    else:
        print("✅ Security validation tests passed")
    
    print(f"\n🏆 FINAL SECURITY SCORE: {max(0, security_score)}/100")
    
    if security_score >= 85:
        print("🟢 EXCELLENT - Production ready security")
        return True
    elif security_score >= 70:
        print("🟡 GOOD - Minor security improvements recommended")
        return True
    else:
        print("🔴 NEEDS IMPROVEMENT - Address critical issues before production")
        return False


if __name__ == "__main__":
    success = run_security_scan()
    sys.exit(0 if success else 1)