"""Test case loading and management."""

import json
from pathlib import Path
from typing import List, Optional

from .models import TestCase


class CaseLoader:
    """Load and manage test cases."""
    
    @staticmethod
    def load_bundled_cases() -> List[TestCase]:
        """Load the standard bundled test cases.
        
        Returns:
            List of all standard test cases
        """
        data_file = Path(__file__).parent.parent.parent / "data" / "test_cases.json"
        return CaseLoader.load_from_file(data_file)
    
    @staticmethod
    def load_from_file(file_path: Path) -> List[TestCase]:
        """Load test cases from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of test cases
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test case file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract cases from the structure
            if "cases" in data:
                cases_data = data["cases"]
            else:
                # Support simple list format too
                cases_data = data if isinstance(data, list) else [data]
            
            return [TestCase(**case_data) for case_data in cases_data]
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading test cases from {file_path}: {e}")
    
    @staticmethod
    def get_case_by_name(name: str, cases: Optional[List[TestCase]] = None) -> Optional[TestCase]:
        """Find a test case by name.
        
        Args:
            name: Case name to find
            cases: List of cases to search (loads bundled if not provided)
            
        Returns:
            TestCase if found, None otherwise
        """
        if cases is None:
            cases = CaseLoader.load_bundled_cases()
        
        for case in cases:
            if case.name == name:
                return case
        return None
    
    @staticmethod
    def get_cases_by_theme(theme: str, cases: Optional[List[TestCase]] = None) -> List[TestCase]:
        """Get all cases matching a theme.
        
        Args:
            theme: Theme to filter by
            cases: List of cases to search (loads bundled if not provided)
            
        Returns:
            List of matching test cases
        """
        if cases is None:
            cases = CaseLoader.load_bundled_cases()
        
        return [case for case in cases if case.theme == theme]
    
    @staticmethod
    def get_explicit_cases(cases: Optional[List[TestCase]] = None) -> List[TestCase]:
        """Get all explicit condition cases.
        
        Args:
            cases: List of cases to filter (loads bundled if not provided)
            
        Returns:
            List of explicit test cases
        """
        if cases is None:
            cases = CaseLoader.load_bundled_cases()
        
        return [case for case in cases if case.condition == "Explicit"]
    
    @staticmethod
    def get_implicit_cases(cases: Optional[List[TestCase]] = None) -> List[TestCase]:
        """Get all implicit condition cases.
        
        Args:
            cases: List of cases to filter (loads bundled if not provided)
            
        Returns:
            List of implicit test cases
        """
        if cases is None:
            cases = CaseLoader.load_bundled_cases()
        
        return [case for case in cases if case.condition == "Implicit"]
