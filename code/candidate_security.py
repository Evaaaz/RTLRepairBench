"""Fail-closed lexical preflight for RTL evaluated inside trusted harnesses.

The simulators and formal tools used by this package compile model-generated
RTL in the same process as the verifier harness.  This scanner keeps that RTL
to one ordinary design module and rejects constructs that can constrain a
proof, inspect or alter harness state, emit authenticated-looking output, stop
the simulator, or load additional source.  It is intentionally conservative:
rejected sources are evaluation failures, never successful repairs.

This is a security boundary, not a complete SystemVerilog parser.  Harnesses
must additionally use unpredictable names/nonces and authenticate terminal
verdicts; callers must not weaken either layer based on this preflight alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


# Side-effect-free elaboration/query functions admitted in otherwise ordinary
# synthesizable RTL.  All other ``$...`` names are rejected.  Keeping an
# allowlist avoids overlooking a simulator-specific output/control/file/VPI
# task while retaining common parameterized designs.
_SAFE_SYSTEM_FUNCTIONS = frozenset(
    {
        "$bits",
        "$clog2",
        "$countbits",
        "$countones",
        "$dimensions",
        "$high",
        "$increment",
        "$isunknown",
        "$left",
        "$low",
        "$onehot",
        "$onehot0",
        "$right",
        "$signed",
        "$size",
        "$typename",
        "$unpacked_dimensions",
        "$unsigned",
    }
)

_FORBIDDEN_PATTERNS = (
    (r"\b(?:assert|assume|cover|restrict)\b", "formal constraint/check statement"),
    (
        r"\b(?:property|endproperty|sequence|endsequence|checker|endchecker|bind|"
        r"interface|endinterface|program|endprogram|primitive|endprimitive|"
        r"package|endpackage|config|endconfig|macromodule)\b",
        "embedded verification or additional design construct",
    ),
    (r"\b(?:force|release)\b", "cross-hierarchy force/release"),
    (r"\$root\b", "root hierarchy access"),
    (
        r"\b[A-Za-z_][A-Za-z0-9_$]*\s*\.\s*[A-Za-z_][A-Za-z0-9_$]*\b",
        "hierarchical or cross-scope reference",
    ),
)

_DRIVING_PRIMITIVES = frozenset(
    {
        "and",
        "buf",
        "bufif0",
        "bufif1",
        "cmos",
        "nand",
        "nmos",
        "nor",
        "not",
        "notif0",
        "notif1",
        "or",
        "pmos",
        "pulldown",
        "pullup",
        "rcmos",
        "rnmos",
        "rpmos",
        "rtran",
        "rtranif0",
        "rtranif1",
        "tran",
        "tranif0",
        "tranif1",
        "xnor",
        "xor",
    }
)
_BIDIRECTIONAL_PRIMITIVES = frozenset(
    {"rtran", "rtranif0", "rtranif1", "tran", "tranif0", "tranif1"}
)
_MULTI_OUTPUT_PRIMITIVES = frozenset({"buf", "not"})
_DRIVE_STRENGTHS = frozenset(
    {
        "highz0",
        "highz1",
        "pull0",
        "pull1",
        "small",
        "strong0",
        "strong1",
        "supply0",
        "supply1",
        "weak0",
        "weak1",
    }
)


def mask_sv_comments_and_strings(source: str) -> str:
    """Mask comments and string contents while preserving token boundaries."""

    masked = list(source)
    index = 0
    state = "code"
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char == "/" and following == "/":
                masked[index] = masked[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if char == "/" and following == "*":
                masked[index] = masked[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                masked[index] = " "
                index += 1
                state = "string"
                continue
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                masked[index] = " "
            index += 1
            continue
        elif state == "block_comment":
            if char == "*" and following == "/":
                masked[index] = masked[index + 1] = " "
                index += 2
                state = "code"
                continue
            if char != "\n":
                masked[index] = " "
            index += 1
            continue
        else:
            if char == "\\" and following:
                masked[index] = " "
                if following != "\n":
                    masked[index + 1] = " "
                index += 2
                continue
            if char == '"':
                masked[index] = " "
                state = "code"
            elif char != "\n":
                masked[index] = " "
            index += 1
            continue
        index += 1
    return "".join(masked)


def candidate_security_reason(
    source: str,
    *,
    reserved_hierarchy: Iterable[str] = (),
) -> str:
    """Return an explanatory rejection reason, or ``""`` for admitted RTL."""

    code = mask_sv_comments_and_strings(source)
    modules = len(re.findall(r"\bmodule\b", code, flags=re.IGNORECASE))
    endmodules = len(re.findall(r"\bendmodule\b", code, flags=re.IGNORECASE))
    if modules > 1 or endmodules > 1 or (modules == 1) != (endmodules == 1):
        return (
            "candidate security preflight requires exactly one module "
            f"(found module={modules}, endmodule={endmodules})"
        )

    # Attributes can alter synthesis/formal semantics (for example black-box or
    # formal signal attributes), so model-generated attributes are outside the
    # audited subset even when a simulator would ignore them.
    without_wildcard_events = re.sub(r"@\s*\(\s*\*\s*\)", " ", code)
    if "(*" in without_wildcard_events or "*)" in without_wildcard_events:
        return "candidate security preflight rejected source attribute"

    # Preprocessor state and included files cross the source-file boundary.
    # No directive is needed by the released benchmark subset.
    if re.search(r"`[A-Za-z_]|``", code):
        return "candidate security preflight rejected preprocessor directive"

    for pattern, description in _FORBIDDEN_PATTERNS:
        if re.search(pattern, code, flags=re.IGNORECASE):
            return f"candidate security preflight rejected {description}"

    system_names = {
        match.group(0).lower()
        for match in re.finditer(r"\$[A-Za-z_][A-Za-z0-9_$]*", code)
    }
    unsafe_system_names = sorted(system_names - _SAFE_SYSTEM_FUNCTIONS)
    if unsafe_system_names:
        return (
            "candidate security preflight rejected system task/function "
            + ", ".join(unsafe_system_names)
        )

    for identifier in reserved_hierarchy:
        if not identifier:
            continue
        if re.search(rf"\b{re.escape(identifier)}\b", code, flags=re.IGNORECASE):
            return "candidate security preflight rejected reserved verifier hierarchy"
    return ""


def input_port_drive_reason(source: str, input_names: Iterable[str]) -> str:
    """Reject candidate constructs that can drive a verifier-owned input net.

    Verilog input nets may be collapsed with the parent harness net.  Letting a
    candidate drive one can therefore constrain both the candidate and golden
    instance, turning an equivalence proof vacuous.  This lightweight check is
    conservative and is applied only after the caller has parsed the interface.
    """

    names = tuple(sorted({name for name in input_names if name}))
    if not names:
        return ""
    code = mask_sv_comments_and_strings(source)
    name_pattern = "(?:" + "|".join(re.escape(name) for name in names) + ")"

    # An ANSI input may not be redeclared as an output/inout in the module body.
    # Some frontends resolve duplicate direction declarations differently; the
    # verifier does not admit that ambiguity at its trust boundary.
    header_terminator = re.search(r"\)\s*;", code)
    if header_terminator is not None and re.search(
        rf"\b(?:input|output|inout)\b[^;]*\b{name_pattern}\b[^;]*;",
        code[header_terminator.end() :],
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return "candidate security preflight rejected input direction redeclaration"

    # Explicit net aliases, procedural continuous assignment controls, and
    # increment/compound-assignment forms are unambiguously candidate drives.
    direct_patterns = (
        rf"\b(?:alias|deassign)\b[^;]*\b{name_pattern}\b",
        rf"(?:\+\+|--)\s*\b{name_pattern}\b",
        rf"\b{name_pattern}\b\s*(?:\+\+|--|[+\-*/%&|^]|<<|>>)=",
        rf"\{{[^{{}};]*\b{name_pattern}\b[^{{}};]*\}}\s*(?:<=|(?<![=!<>])=(?!=))",
    )
    for pattern in direct_patterns:
        if re.search(pattern, code, flags=re.IGNORECASE | re.DOTALL):
            return "candidate security preflight rejected drive of input port"

    # A blocking assignment is never a comparison.  Permit selects on the LHS
    # but do not consume an equality operator.
    blocking = re.compile(
        rf"\b{name_pattern}\b(?:\s*\[[^\]]+\])*\s*(?<![=!<>])=(?!=)",
        flags=re.IGNORECASE,
    )
    if blocking.search(code):
        return "candidate security preflight rejected drive of input port"

    # ``<=`` is both nonblocking assignment and comparison.  It is a comparison
    # inside a control predicate or after an earlier assignment operator in the
    # same statement; otherwise a statement-position occurrence is a drive.
    nonblocking = re.compile(
        rf"\b{name_pattern}\b(?:\s*\[[^\]]+\])*\s*<=",
        flags=re.IGNORECASE,
    )
    for match in nonblocking.finditer(code):
        prefix = code[: match.start()]
        boundary = max(prefix.rfind(";"), prefix.rfind("begin"), prefix.rfind("end"))
        statement_prefix = prefix[boundary + 1 :]
        if re.search(r"(?<![=!<>])=(?!=)|<=", statement_prefix):
            continue
        # Track the innermost open parenthesis and classify common expression
        # contexts.  A bare ``a <= value;`` remains a nonblocking drive.
        open_index = prefix.rfind("(")
        close_index = prefix.rfind(")")
        if open_index > close_index:
            before_open = prefix[:open_index]
            keyword = re.search(r"([A-Za-z_][A-Za-z0-9_$]*)\s*$", before_open)
            if keyword and keyword.group(1).lower() in {
                "if",
                "while",
                "repeat",
                "wait",
                "case",
                "casex",
                "casez",
            }:
                continue
            # Parenthesized RHS/expression, rather than a statement LHS.
            if "=" in statement_prefix or "?" in statement_prefix:
                continue
        prior = statement_prefix.rstrip()
        if prior and prior[-1] not in ")":
            # Operators, commas, and opening parentheses put the input in an
            # expression rather than at the start of an assignment statement.
            if prior[-1] in "(,+-*/%&|^!~?:<>":
                continue
        return "candidate security preflight rejected drive of input port"

    # Built-in gates drive their first terminal; tran-family terminals are all
    # bidirectional.  Reject only when a verifier-owned input occupies a driven
    # terminal, so ordinary ``and (out, in1, in2)`` remains admissible.
    primitive_re = re.compile(
        r"(?:^|;|\bbegin\b(?:\s*:\s*[A-Za-z_][A-Za-z0-9_$]*)?|\bgenerate\b)"
        r"\s*(" + "|".join(sorted(_DRIVING_PRIMITIVES)) + r")\b([^;]*);",
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    for primitive_match in primitive_re.finditer(code):
        primitive = primitive_match.group(1).lower()
        body = primitive_match.group(2)
        groups: list[tuple[int, str]] = []
        depth = 0
        start = -1
        for index, char in enumerate(body):
            if char == "(":
                if depth == 0:
                    start = index
                depth += 1
            elif char == ")" and depth:
                depth -= 1
                if depth == 0 and start >= 0:
                    groups.append((start, body[start + 1 : index]))

        for start, group in groups:
            preceding = body[:start].rstrip()
            if preceding.endswith("#"):
                continue  # gate delay, not an instance terminal list
            terminals: list[str] = []
            terminal_start = 0
            nested = 0
            for index, char in enumerate(group):
                if char in "([{":
                    nested += 1
                elif char in ")]}" and nested:
                    nested -= 1
                elif char == "," and nested == 0:
                    terminals.append(group[terminal_start:index].strip())
                    terminal_start = index + 1
            terminals.append(group[terminal_start:].strip())
            normalized = {part.lower() for part in terminals}
            if terminals and normalized <= _DRIVE_STRENGTHS:
                continue
            if primitive in _MULTI_OUTPUT_PRIMITIVES:
                driven = terminals[:-1]
            elif primitive in _BIDIRECTIONAL_PRIMITIVES:
                driven = terminals[:2]
            else:
                driven = terminals[:1]
            for terminal in driven:
                if re.search(rf"\b{name_pattern}\b", terminal, flags=re.IGNORECASE):
                    return (
                        "candidate security preflight rejected primitive drive "
                        "of input port"
                    )
    return ""
