"""Test case loading and management."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .models import TestCase


class CaseLoader:
    """Load and manage test cases."""

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    
    @staticmethod
    def load_bundled_cases() -> List[TestCase]:
        """Load the standard bundled test cases.
        
        Returns:
            List of all standard test cases
        """
        data_file = CaseLoader.DATA_DIR / "test_cases.json"
        return CaseLoader.load_from_file(data_file)

    @staticmethod
    def load_bundled_language_cases(language: str) -> List[TestCase]:
        """Load bundled cases for a specific language.

        English cases are loaded from the original bundled benchmark. Other
        languages use files named test_cases_<language>.json, for example
        data/test_cases_pl.json.
        """
        normalized = language.lower()
        if normalized == "en":
            return CaseLoader.load_bundled_cases()

        data_file = CaseLoader.DATA_DIR / f"test_cases_{normalized}.json"
        return CaseLoader.load_from_file(data_file)

    @staticmethod
    def load_bundled_multilingual_cases(languages: Optional[List[str]] = None) -> List[TestCase]:
        """Load bundled cases for multiple languages.

        Args:
            languages: Language tags to load. Defaults to English plus every
                available data/test_cases_<language>.json file.

        Returns:
            Combined list of test cases.
        """
        if languages is None:
            language_files = sorted(CaseLoader.DATA_DIR.glob("test_cases_*.json"))
            languages = ["en"] + [
                path.stem.removeprefix("test_cases_")
                for path in language_files
            ]

        cases: List[TestCase] = []
        for language in languages:
            cases.extend(CaseLoader.load_bundled_language_cases(language))
        return cases
    
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
    def get_cases_by_language(language: str, cases: Optional[List[TestCase]] = None) -> List[TestCase]:
        """Get all cases matching a language tag."""
        if cases is None:
            cases = CaseLoader.load_bundled_multilingual_cases()

        return [case for case in cases if case.language.lower() == language.lower()]

    @staticmethod
    def get_paired_cases(
        cases: Optional[List[TestCase]] = None,
        languages: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, TestCase]]:
        """Group translated cases by canonical source id and language.

        The source id is source_case_id when present, otherwise the case id.
        Only groups containing all requested languages are returned.
        """
        if cases is None:
            cases = CaseLoader.load_bundled_multilingual_cases(languages)

        requested = [language.lower() for language in languages] if languages else None
        paired: Dict[str, Dict[str, TestCase]] = {}
        for case in cases:
            source_id = case.source_case_id or case.id
            paired.setdefault(source_id, {})[case.language.lower()] = case

        if requested:
            paired = {
                source_id: by_language
                for source_id, by_language in paired.items()
                if all(language in by_language for language in requested)
            }

        return paired
    
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
