"""
Mathematical Expression Formatter for GuruAI

This module provides utilities to format mathematical expressions in responses
to ensure proper LaTeX rendering on the frontend.

Features:
- Detect mathematical expressions in text
- Wrap expressions with LaTeX delimiters
- Format common mathematical patterns
- Preserve existing LaTeX formatting

Requirements: 6.5
"""

import re
from typing import List, Tuple


class MathFormatter:
    """
    Formats mathematical expressions for LaTeX rendering.
    
    Handles both inline ($...$) and display ($$...$$) math modes.
    """
    
    # Common mathematical patterns
    MATH_PATTERNS = [
        # Fractions: a/b -> \frac{a}{b}
        (r'(\d+)/(\d+)', r'\\frac{\1}{\2}'),
        
        # Superscripts: x^2 -> x^{2}
        (r'(\w)\^(\d+)', r'\1^{\2}'),
        
        # Subscripts: x_1 -> x_{1}
        (r'(\w)_(\d+)', r'\1_{\2}'),
        
        # Square roots: sqrt(x) -> \sqrt{x}
        (r'sqrt\(([^)]+)\)', r'\\sqrt{\1}'),
        
        # Greek letters (common ones)
        (r'\balpha\b', r'\\alpha'),
        (r'\bbeta\b', r'\\beta'),
        (r'\bgamma\b', r'\\gamma'),
        (r'\bdelta\b', r'\\delta'),
        (r'\btheta\b', r'\\theta'),
        (r'\bpi\b', r'\\pi'),
        (r'\bsigma\b', r'\\sigma'),
        (r'\bomega\b', r'\\omega'),
    ]
    
    @staticmethod
    def has_latex_delimiters(text: str) -> bool:
        """
        Check if text already contains LaTeX delimiters.
        
        Args:
            text: Text to check
        
        Returns:
            True if LaTeX delimiters are present
        """
        if not text:
            return False
        
        # Check for common LaTeX delimiters (simple string checks)
        if '$$' in text or '$' in text:
            return True
        
        if '\\[' in text or '\\(' in text:
            return True
        
        return False
    
    @staticmethod
    def wrap_inline_math(expression: str) -> str:
        """
        Wrap expression with inline math delimiters.
        
        Args:
            expression: Mathematical expression
        
        Returns:
            Expression wrapped with $...$
        """
        # Remove existing delimiters if present
        expression = expression.strip()
        if expression.startswith('$') and expression.endswith('$'):
            return expression
        
        return f'${expression}$'
    
    @staticmethod
    def wrap_display_math(expression: str) -> str:
        """
        Wrap expression with display math delimiters.
        
        Args:
            expression: Mathematical expression
        
        Returns:
            Expression wrapped with $$...$$
        """
        # Remove existing delimiters if present
        expression = expression.strip()
        if expression.startswith('$$') and expression.endswith('$$'):
            return expression
        
        return f'$${expression}$$'
    
    @staticmethod
    def format_equation(equation: str, display: bool = False) -> str:
        """
        Format an equation with proper LaTeX syntax.
        
        Args:
            equation: Equation string (e.g., "F = ma")
            display: Whether to use display mode
        
        Returns:
            Formatted equation with LaTeX delimiters
        """
        if not equation:
            return equation
        
        # Skip if already has LaTeX delimiters
        if MathFormatter.has_latex_delimiters(equation):
            return equation
        
        # Apply common patterns
        formatted = equation
        for pattern, replacement in MathFormatter.MATH_PATTERNS:
            formatted = re.sub(pattern, replacement, formatted)
        
        # Wrap with appropriate delimiters
        if display:
            return MathFormatter.wrap_display_math(formatted)
        else:
            return MathFormatter.wrap_inline_math(formatted)
    
    @staticmethod
    def detect_math_expressions(text: str) -> List[Tuple[int, int, str]]:
        """
        Detect potential mathematical expressions in text.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of tuples (start_pos, end_pos, expression)
        """
        expressions = []
        
        # Pattern for equations (contains =, +, -, *, /, ^, etc.)
        equation_pattern = r'\b[a-zA-Z0-9\s\+\-\*/\^=\(\)]+\b'
        
        for match in re.finditer(equation_pattern, text):
            expr = match.group(0)
            
            # Check if it looks like math (has operators)
            if any(op in expr for op in ['=', '+', '-', '*', '/', '^']):
                # Check if it has variables or numbers
                if re.search(r'[a-zA-Z]', expr) or re.search(r'\d', expr):
                    expressions.append((match.start(), match.end(), expr))
        
        return expressions
    
    @staticmethod
    def format_text_with_math(text: str, auto_detect: bool = False) -> str:
        """
        Format text containing mathematical expressions.
        
        Args:
            text: Text with potential math expressions
            auto_detect: Whether to auto-detect and format expressions
        
        Returns:
            Text with properly formatted math expressions
        """
        if not text:
            return text
        
        # If already has LaTeX, return as is
        if MathFormatter.has_latex_delimiters(text):
            return text
        
        # If auto-detect is disabled, return as is
        if not auto_detect:
            return text
        
        # Detect and format expressions
        expressions = MathFormatter.detect_math_expressions(text)
        
        # Sort by position (reverse order to maintain indices)
        expressions.sort(key=lambda x: x[0], reverse=True)
        
        # Replace expressions with formatted versions
        formatted_text = text
        for start, end, expr in expressions:
            formatted_expr = MathFormatter.format_equation(expr, display=False)
            formatted_text = formatted_text[:start] + formatted_expr + formatted_text[end:]
        
        return formatted_text
    
    @staticmethod
    def format_formula(formula: str, name: str = None) -> str:
        """
        Format a formula with optional name.
        
        Args:
            formula: Formula expression
            name: Optional formula name
        
        Returns:
            Formatted formula string
        """
        formatted_formula = MathFormatter.format_equation(formula, display=True)
        
        if name:
            return f"**{name}:**\n{formatted_formula}"
        
        return formatted_formula
    
    @staticmethod
    def format_step_with_math(step_text: str) -> str:
        """
        Format a solution step that may contain math.
        
        Args:
            step_text: Step description
        
        Returns:
            Formatted step with proper math rendering
        """
        # Common patterns in solution steps
        # Example: "Substitute x = 5" -> "Substitute $x = 5$"
        
        # Pattern for simple equations in text
        equation_in_text = r'([a-zA-Z]\s*=\s*[0-9\.\-]+)'
        
        def replace_equation(match):
            return MathFormatter.wrap_inline_math(match.group(1))
        
        formatted = re.sub(equation_in_text, replace_equation, step_text)
        
        return formatted


def format_response_with_math(response: str) -> str:
    """
    Format a complete response to ensure math expressions are properly formatted.
    
    This is a convenience function that can be used to post-process LLM responses.
    
    Args:
        response: Response text from LLM
    
    Returns:
        Response with properly formatted math expressions
    """
    # If response already has LaTeX, return as is
    if MathFormatter.has_latex_delimiters(response):
        return response
    
    # Otherwise, return as is (let frontend handle it)
    # We don't want to be too aggressive with auto-formatting
    return response


def main():
    """Test the math formatter."""
    print("="*60)
    print("Math Formatter Test")
    print("="*60)
    
    # Test cases
    test_cases = [
        ("F = ma", "Simple equation"),
        ("x^2 + 2x + 1 = 0", "Quadratic equation"),
        ("E = mc^2", "Einstein's equation"),
        ("The formula is F = ma where F is force", "Equation in text"),
        ("sqrt(x^2 + y^2)", "Square root"),
        ("The angle theta is important", "Greek letter"),
        ("$x = 5$", "Already formatted"),
    ]
    
    print("\nTest Cases:")
    print("-"*60)
    
    for expr, description in test_cases:
        formatted = MathFormatter.format_equation(expr)
        print(f"\n{description}:")
        print(f"  Input:  {expr}")
        print(f"  Output: {formatted}")
    
    # Test formula formatting
    print("\n" + "="*60)
    print("Formula Formatting")
    print("="*60)
    
    formula = "a^2 + b^2 = c^2"
    formatted_formula = MathFormatter.format_formula(formula, "Pythagorean Theorem")
    print(f"\n{formatted_formula}")
    
    # Test step formatting
    print("\n" + "="*60)
    print("Step Formatting")
    print("="*60)
    
    step = "Substitute x = 5 into the equation"
    formatted_step = MathFormatter.format_step_with_math(step)
    print(f"\nOriginal: {step}")
    print(f"Formatted: {formatted_step}")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)


if __name__ == "__main__":
    main()
