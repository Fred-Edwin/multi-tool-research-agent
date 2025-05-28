"""Calculator tool for mathematical computations."""

import re
import math
from typing import Union, Dict, Any
from .base_tool import BaseTool, ToolInput, ToolOutput


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations and computations"
        )
        # Safe mathematical functions
        self.safe_functions = {
            'abs': abs,
            'round': round,
            'max': max,
            'min': min,
            'sum': sum,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e,
        }
    
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """Execute mathematical calculation."""
        try:
            query = input_data.query.strip()
            if not query:
                return self._create_error_output("Empty calculation query provided")
            
            # Extract mathematical expression from query
            expression = self._extract_expression(query)
            if not expression:
                return self._create_error_output("No mathematical expression found in query")
            
            # Perform calculation
            result = self._calculate(expression)
            
            # Format result
            formatted_result = self._format_result(query, expression, result)
            
            return ToolOutput(
                result=formatted_result,
                source="calculator",
                confidence=0.95,
                metadata={
                    "original_query": query,
                    "expression": expression,
                    "numeric_result": result
                }
            )
            
        except Exception as e:
            self.logger.error(f"Calculator error: {str(e)}")
            return self._create_error_output(f"Calculation failed: {str(e)}")
    
    def _extract_expression(self, query: str) -> str:
        """Extract mathematical expression from natural language query."""
        # Handle compound interest calculation
        if "compound interest" in query.lower():
            return self._extract_compound_interest(query)
        
        # Handle percentage calculations
        if "%" in query or "percent" in query.lower():
            return self._extract_percentage(query)
        
        # Extract basic mathematical expressions
        # Look for patterns like "calculate X", "what is X", etc.
        patterns = [
            r"calculate\s+(.+)",
            r"what\s+is\s+(.+)",
            r"compute\s+(.+)",
            r"solve\s+(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                expr = match.group(1).strip()
                # Clean up the expression
                expr = self._clean_expression(expr)
                return expr
        
        # If no pattern matches, try to find mathematical expressions directly
        math_pattern = r'[\d+\-*/().\s]+'
        matches = re.findall(math_pattern, query)
        if matches:
            return max(matches, key=len).strip()
        
        return ""
    
    def _extract_compound_interest(self, query: str) -> str:
        """Extract compound interest calculation parameters."""
        try:
            # Extract principal, rate, and time
            principal_match = re.search(r'\$?([\d,]+)', query)
            rate_match = re.search(r'(\d+(?:\.\d+)?)%', query)
            time_match = re.search(r'(\d+)\s*years?', query)
            
            if principal_match and rate_match and time_match:
                principal = float(principal_match.group(1).replace(',', ''))
                rate = float(rate_match.group(1)) / 100
                time = float(time_match.group(1))
                
                # Compound interest formula: A = P(1 + r)^t
                return f"{principal} * (1 + {rate}) ** {time}"
            
        except Exception:
            pass
        
        return ""
    
    def _extract_percentage(self, query: str) -> str:
        """Extract percentage calculations."""
        try:
            # Pattern: X% of Y
            pattern = r'(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:,\d{3})*(?:\.\d+)?)'
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                percentage = float(match.group(1))
                number = float(match.group(2).replace(',', ''))
                return f"({percentage} / 100) * {number}"
            
        except Exception:
            pass
        
        return ""
    
    def _clean_expression(self, expr: str) -> str:
        """Clean and validate mathematical expression."""
        # Remove common words that might interfere
        words_to_remove = ['equals', 'equal', 'is', 'the', 'result', 'answer']
        for word in words_to_remove:
            expr = re.sub(rf'\b{word}\b', '', expr, flags=re.IGNORECASE)
        
        # Replace common mathematical terms
        replacements = {
            'times': '*',
            'multiplied by': '*',
            'divided by': '/',
            'plus': '+',
            'minus': '-',
            'squared': '**2',
            'cubed': '**3',
        }
        
        for term, replacement in replacements.items():
            expr = re.sub(rf'\b{term}\b', replacement, expr, flags=re.IGNORECASE)
        
        return expr.strip()
    
    def _calculate(self, expression: str) -> Union[float, int]:
        """Safely evaluate mathematical expression."""
        try:
            # Create a safe evaluation environment
            safe_dict = {
                "__builtins__": {},
                **self.safe_functions
            }
            
            # Evaluate the expression
            result = eval(expression, safe_dict)
            
            # Return int if it's a whole number, float otherwise
            if isinstance(result, float) and result.is_integer():
                return int(result)
            return result
            
        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {expression}")
    
    def _format_result(self, original_query: str, expression: str, result: Union[float, int]) -> str:
        """Format the calculation result."""
        formatted_result = f"Calculation: {expression} = {result:,}"
        
        # Add context if it was a compound interest calculation
        if "compound interest" in original_query.lower():
            formatted_result += f"\n\nThe compound interest calculation shows that the final amount would be ${result:,.2f}"
        
        # Add context for percentage calculations
        elif "%" in original_query:
            formatted_result += f"\n\nThe percentage calculation result is {result:,}"
        
        return formatted_result
    
    def is_relevant(self, query: str) -> bool:
        """Check if calculator is relevant for the query."""
        calc_keywords = [
            "calculate", "compute", "math", "mathematical", "equation",
            "add", "subtract", "multiply", "divide", "sum", "total",
            "percentage", "percent", "interest", "compound", "+", "-", "*", "/",
            "what is", "how much", "equals"
        ]
        
        query_lower = query.lower()
        has_numbers = bool(re.search(r'\d', query))
        has_keywords = any(keyword in query_lower for keyword in calc_keywords)
        
        return has_numbers and has_keywords