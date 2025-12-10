# File: tools/purity/messages.py
"""Error message templates for purity violations.

Provides detailed messages with before/after examples for each PUR rule.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleMessage:
    """Error message template for a purity rule."""

    code: str
    short_message: str
    long_message: str
    before_example: str
    after_example: str
    doc_link: str


# PUR001: No For Loops in Business Logic
PUR001_MESSAGE = RuleMessage(
    code="PUR001",
    short_message="for loop forbidden in business logic",
    long_message=(
        "For loops are forbidden in Tier 2 business logic files.\n"
        "Use list/dict/set comprehensions, generator expressions, or Effect sequences instead.\n"
        "For loops are allowed in Tier 1 infrastructure and Tier 3 effect interpreters."
    ),
    before_example=(
        "# ❌ FORBIDDEN in Tier 2\n"
        "results = []\n"
        "for item in items:\n"
        "    results.append(process(item))\n"
        "return results"
    ),
    after_example=("# ✅ CORRECT - Use comprehension\n" "return [process(item) for item in items]"),
    doc_link="documents/engineering/purity_enforcement.md#rule-pur001",
)

# PUR002: No While Loops in Business Logic
PUR002_MESSAGE = RuleMessage(
    code="PUR002",
    short_message="while loop forbidden in business logic",
    long_message=(
        "While loops are forbidden in Tier 2 business logic files.\n"
        "Use generators, itertools, or recursive generators instead.\n"
        "While loops are allowed in Tier 1 infrastructure and Tier 3 effect interpreters."
    ),
    before_example=(
        "# ❌ FORBIDDEN in Tier 2\n"
        "while condition:\n"
        "    do_something()\n"
        "    update_condition()"
    ),
    after_example=(
        "# ✅ CORRECT - Use generator\n"
        "def generate_until_done() -> Generator[T, None, None]:\n"
        "    yield from takewhile(lambda x: condition(x), stream)"
    ),
    doc_link="documents/engineering/purity_enforcement.md#rule-pur002",
)

# PUR003: No If Statements in Business Logic
PUR003_MESSAGE = RuleMessage(
    code="PUR003",
    short_message="if statement forbidden in business logic",
    long_message=(
        "Statement-level if branches are forbidden in Tier 2 business logic files.\n"
        "Use conditional expressions (ternary), match/case, or filter comprehensions instead.\n"
        "If statements are allowed in Tier 1 infrastructure and Tier 3 effect interpreters.\n"
        "Note: 4 boundary exceptions are whitelisted (see purity_enforcement.md)."
    ),
    before_example=(
        "# ❌ FORBIDDEN in Tier 2\n"
        "if condition:\n"
        "    result = value_a\n"
        "else:\n"
        "    result = value_b\n"
        "return result"
    ),
    after_example=(
        "# ✅ CORRECT - Use conditional expression\n"
        "return value_a if condition else value_b\n\n"
        "# ✅ CORRECT - Use match/case for complex branching\n"
        "match value:\n"
        "    case Pattern1():\n"
        "        return result1\n"
        "    case Pattern2():\n"
        "        return result2"
    ),
    doc_link="documents/engineering/purity_enforcement.md#rule-pur003",
)

# PUR004: No Raise for Expected Errors
PUR004_MESSAGE = RuleMessage(
    code="PUR004",
    short_message="raise forbidden for expected errors in business logic",
    long_message=(
        "Raising exceptions for expected errors is forbidden in Tier 2 business logic.\n"
        "Use Result[T, E] types instead to model errors as data.\n"
        "Exceptions are only allowed for:\n"
        "  - Programming errors (assertions, unreachable code)\n"
        "  - Inside assert_never() calls\n"
        "  - Tier 1 infrastructure and Tier 3 effect interpreters"
    ),
    before_example=(
        "# ❌ FORBIDDEN in Tier 2\n"
        "def load_checkpoint(path: Path) -> Checkpoint:\n"
        "    if not path.exists():\n"
        "        raise FileNotFoundError(f'Not found: {path}')\n"
        "    return deserialize(path.read_bytes())"
    ),
    after_example=(
        "# ✅ CORRECT - Use Result type\n"
        "def load_checkpoint(path: Path) -> Result[Checkpoint, LoadError]:\n"
        "    return (\n"
        "        Failure(CheckpointNotFound(path=path))\n"
        "        if not path.exists()\n"
        "        else Success(deserialize(path.read_bytes()))\n"
        "    )"
    ),
    doc_link="documents/engineering/purity_enforcement.md#rule-pur004",
)

# PUR005: No Side Effects in Business Logic
PUR005_MESSAGE = RuleMessage(
    code="PUR005",
    short_message="side effect forbidden in business logic",
    long_message=(
        "Direct side effects are forbidden in Tier 2 business logic.\n"
        "Model all I/O and logging as Effect ADTs; interpreters execute them.\n"
        "Forbidden calls: print(), logger.info/debug/warning/error(), etc.\n"
        "Side effects are allowed in Tier 1 infrastructure and Tier 3 effect interpreters."
    ),
    before_example=(
        "# ❌ FORBIDDEN in Tier 2\n"
        "def train_step(model: CVNN) -> None:\n"
        "    loss = compute_loss(model)\n"
        "    logger.info(f'Loss: {loss}')\n"
        "    update_weights(model)"
    ),
    after_example=(
        "# ✅ CORRECT - Use Effect ADT\n"
        "def train_step(model: CVNN) -> tuple[CVNN, list[Effect]]:\n"
        "    loss = compute_loss(model)\n"
        "    effects = [LogMessage(level='info', message=f'Loss: {loss}')]\n"
        "    updated_model = update_weights(model)\n"
        "    return (updated_model, effects)"
    ),
    doc_link="documents/engineering/purity_enforcement.md#rule-pur005",
)

# Mapping from rule code to message
RULE_MESSAGES: dict[str, RuleMessage] = {
    "PUR001": PUR001_MESSAGE,
    "PUR002": PUR002_MESSAGE,
    "PUR003": PUR003_MESSAGE,
    "PUR004": PUR004_MESSAGE,
    "PUR005": PUR005_MESSAGE,
}


def format_violation(
    rule_code: str,
    filepath: str,
    line_number: int,
    context: str = "",
    verbose: bool = False,
) -> str:
    """Format a purity violation message.

    Args:
        rule_code: Rule code (PUR001, PUR002, etc.)
        filepath: Path to file with violation
        line_number: Line number of violation
        context: Optional code context
        verbose: If True, include long message and examples

    Returns:
        Formatted error message string
    """
    message = RULE_MESSAGES.get(rule_code)
    if not message:
        return f"{filepath}:{line_number} - Unknown rule: {rule_code}"

    # Short format: file:line - message (code)
    output = f"{filepath}:{line_number} - {message.short_message} ({rule_code})"

    if context:
        output += f"\n  Code: {context}"

    if verbose:
        output += f"\n\n{message.long_message}"
        output += f"\n\nBefore:\n{message.before_example}"
        output += f"\n\nAfter:\n{message.after_example}"
        output += f"\n\nSee: {message.doc_link}"

    return output


def explain_rule(rule_code: str) -> str:
    """Get detailed explanation for a purity rule.

    Args:
        rule_code: Rule code (PUR001, PUR002, etc.)

    Returns:
        Detailed explanation with examples
    """
    message = RULE_MESSAGES.get(rule_code)
    if not message:
        return f"Unknown rule: {rule_code}"

    return (
        f"Rule {message.code}: {message.short_message}\n"
        f"\n{message.long_message}"
        f"\n\nBefore:\n{message.before_example}"
        f"\n\nAfter:\n{message.after_example}"
        f"\n\nDocumentation: {message.doc_link}"
    )
