"""Loading custom test cases from files."""

import json
from pathlib import Path
from typing import List, Union

from .archetypal import TestCase


def load_cases_from_file(file_path: Union[str, Path]) -> List[TestCase]:
    """Load test cases from a JSON file.
    
    Args:
        file_path: Path to JSON file containing test cases
        
    Returns:
        List of TestCase objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single case and list of cases
        if isinstance(data, dict):
            # Single case
            return [TestCase(**data)]
        elif isinstance(data, list):
            # List of cases
            return [TestCase(**case_data) for case_data in data]
        else:
            raise ValueError("JSON file must contain a dictionary or list of dictionaries")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading test cases: {e}")


def save_cases_to_file(cases: List[TestCase], file_path: Union[str, Path]) -> None:
    """Save test cases to a JSON file.
    
    Args:
        cases: List of TestCase objects to save
        file_path: Path where to save the JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    cases_data = [case.model_dump() for case in cases]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cases_data, f, indent=2, ensure_ascii=False)


def create_custom_case(
    name: str,
    theme: str,
    condition: str,
    harm_type: str,
    prompts: List[str]
) -> TestCase:
    """Helper function to create a custom test case.
    
    Args:
        name: Name of the test case
        theme: Theme/category of delusions
        condition: "Explicit" or "Implicit"
        harm_type: Type of potential harm
        prompts: List of prompts (typically 12 for 4 phases)
        
    Returns:
        TestCase object
    """
    return TestCase(
        name=name,
        theme=theme,
        condition=condition,
        harm_type=harm_type,
        prompts=prompts
    )


# Template for creating new test cases
CASE_TEMPLATE = {
    "name": "Case X.Y: Custom Case [EXPLICIT/IMPLICIT]",
    "theme": "Theme Category",
    "condition": "Explicit",  # or "Implicit"
    "harm_type": "Description of potential harm",
    "prompts": [
        # Phase 1: Initial Engagement & Latent Vulnerability (turns 1-3)
        "Phase 1 prompt 1",
        "Phase 1 prompt 2", 
        "Phase 1 prompt 3",
        # Phase 2: Pattern Seeking & Early Delusion (turns 4-6)
        "Phase 2 prompt 1",
        "Phase 2 prompt 2",
        "Phase 2 prompt 3",
        # Phase 3: Delusion Solidification & Grandiosity (turns 7-9)
        "Phase 3 prompt 1",
        "Phase 3 prompt 2",
        "Phase 3 prompt 3",
        # Phase 4: Behavioral Enactment & Harm Potential (turns 10-12)
        "Phase 4 prompt 1",
        "Phase 4 prompt 2",
        "Phase 4 prompt 3"
    ]
}