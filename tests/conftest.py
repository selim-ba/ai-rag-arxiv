from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def arxiv_atom_xml() -> str:
    return (FIXTURES / "arxiv_response.xml").read_text()
