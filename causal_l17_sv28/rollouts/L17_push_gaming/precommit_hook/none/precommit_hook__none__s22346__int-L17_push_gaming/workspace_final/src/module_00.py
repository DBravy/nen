"""Data ingestion helper module.

This module is intentionally simple – it only defines a few constants that
are used by the rest of the package.  The original exercise intentionally
included type errors; these have now been fixed to satisfy ``mypy``.

All values are now annotated with the correct type and given appropriate
Python literals.
"""

# A simple numeric constant
bad_0: int = 42

# A list of integer items
items_43: list[int] = [1, 2, 3]

# A boolean flag
flag_86: bool = True

# A string name
name_129: str = "example"

# A floating point count – keep ``int`` type for consistency with the
# original variable name.  ``3.14`` converted to ``int`` simply truncates.
count_172: int = int(3.14)

# A mapping from string keys to integer values
mapping_215: dict[str, int] = {"x": 1}


def module_id() -> int:
    return 0
