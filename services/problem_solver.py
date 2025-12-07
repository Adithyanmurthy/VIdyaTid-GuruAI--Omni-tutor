"""
Problem Solver - Generate step-by-step solutions for problems from images.

This module processes uploaded images containing problems and generates
detailed, step-by-step solutions with NCERT formula references.

Features:
- Extract problem text from images using OCR
- Generate step-by-step solutions using LLM
- Format solutions with proper mathematical notation
- Add NCERT formula references to each step
- Handle poor quality images with user feedback

Requirements: 2.4, 2.5, 6.1, 6.2
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from services.image_processor import ImageProcessor
from services.model_manager import ModelManager
from services.rag_system import RAGSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProblemSolver:
    """
    Solves problems from uploaded images with step-by-step explanations.
    
    Integrates:
    - Image processing for text extraction
    - RAG system for NCERT context retrieval
    - LLM for solution generation
    - Mathematical notation formatting
    """
    
    def __init__(
        self,
        image_processor: ImageProcessor,
        model_manager: ModelManager,
        rag_system: RAGSystem
    ):
        """
        Initialize the Problem Solver.
        
        Args:
            image_processor: ImageProcessor instance for OCR
            model_manager: ModelManager instance for LLM inference
            rag_system: RAGSystem instance for context retrieval
        """
        self.image_processor = image_processor
        self.llm = model_manager
        self.rag = rag_system
        
        logger.info("ProblemSolver initialized successfully")
    
    async def solve_problem_from_image(
        self,
        image_data: bytes,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an image containing a problem and generate a solution.
        
        Main entry point for problem solving. Performs:
        1. Image validation
        2. Text extraction using OCR
        3. Problem text validation
        4. Solution generation with step-by-step breakdown
        5. NCERT formula reference addition
        
        Args:
            image_data: Raw image bytes
            user_id: Optional user identifier
        
        Returns:
            Dictionary containing:
                - success: Whether processing succeeded
                - problem_text: Extracted problem text
                - solution: Complete solution with steps
                - steps: List of individual solution steps
                - formulas: NCERT formulas referenced
                - diagrams: Relevant diagrams (if any)
                - confidence: Confidence in OCR extraction
                - error: Error message if failed
        """
        try:
            logger.info("Processing problem from image...")
            
            # Step 1: Validate image
            is_valid, error_msg = self.image_processor.validate_image(image_data)
            if not is_valid:
                return self._format_error_response(
                    "Invalid image",
                    error_msg,
                    suggestion="Please upload a clearer image with better quality."
                )
            
            # Step 2: Process image and extract text
            image_result = await self.image_processor.process_image(image_data)
            
            problem_text = image_result.get('text', '')
            confidence = image_result.get('confidence', 'low')
            
            # Check if text extraction was successful
            if not problem_text or len(problem_text.strip()) < 10:
                return self._format_error_response(
                    "Could not extract problem text",
                    "The image quality may be too low or the text is not clear enough.",
                    suggestion="Please upload a clearer image with better lighting and focus."
                )
            
            logger.info(f"Extracted problem text ({len(problem_text)} chars, confidence: {confidence})")
            
            # Step 3: Validate that it's actually a problem
            if not self._is_problem(problem_text):
                return self._format_error_response(
                    "No problem detected",
                    "The extracted text does not appear to contain a problem or question.",
                    suggestion="Please ensure the image contains a clear problem statement."
                )
            
            # Step 4: Retrieve relevant NCERT context
            context_data = self.rag.get_context_for_llm(
                problem_text,
                top_k=5,
                include_references=True
            )
            
            # Step 5: Generate step-by-step solution
            solution_result = await self._generate_solution(
                problem_text,
                context_data
            )
            
            if not solution_result.get('success'):
                return self._format_error_response(
                    "Failed to generate solution",
                    solution_result.get('error', 'Unknown error')
                )
            
            # Step 5.5: Add quality warning if solution seems too short or generic
            solution_text = solution_result['text']
            if len(solution_text) < 200:
                logger.warning(f"Generated solution is very short ({len(solution_text)} chars), may be low quality")
            
            # Step 6: Parse solution into steps
            steps = self._parse_solution_steps(solution_text)
            
            # Step 7: Extract formulas and add NCERT references
            formulas = self._extract_formulas(solution_result['text'], context_data)
            
            # Step 8: Get relevant diagrams
            diagrams = []
            if image_result.get('matched_diagram'):
                diagrams.append(image_result['matched_diagram'])
            
            # Format complete response
            response = {
                'success': True,
                'problem_text': problem_text,
                'solution': solution_result['text'],
                'steps': steps,
                'formulas': formulas,
                'diagrams': diagrams,
                'references': context_data.get('references', []),
                'confidence': confidence,
                'metadata': {
                    'ocr_confidence': confidence,
                    'num_steps': len(steps),
                    'num_formulas': len(formulas),
                    'tokens_used': solution_result.get('tokens_used', 0)
                }
            }
            
            logger.info(f"Problem solved successfully ({len(steps)} steps, {len(formulas)} formulas)")
            return response
            
        except Exception as e:
            logger.error(f"Error solving problem from image: {e}", exc_info=True)
            return self._format_error_response(
                "Problem solving failed",
                str(e)
            )
    
    def _is_problem(self, text: str) -> bool:
        """
        Check if the extracted text contains a problem or question.
        
        Args:
            text: Extracted text from image
        
        Returns:
            True if text appears to be a problem, False otherwise
        """
        # Problem indicators
        problem_indicators = [
            'solve', 'find', 'calculate', 'prove', 'show that',
            'determine', 'compute', 'evaluate', 'derive',
            'question', 'problem', 'exercise', '?',
            'what', 'how', 'why', 'when', 'where',
            'if', 'given', 'a particle', 'a body', 'a system',
            '=', '+', '-', '×', '÷', 'cm', 'kg', 'm/s',
            'accelerate', 'velocity', 'force', 'mass', 'energy'
        ]
        
        text_lower = text.lower()
        
        # Check for indicators
        indicator_count = sum(1 for indicator in problem_indicators if indicator in text_lower)
        
        # Need at least 2 indicators to be confident it's a problem
        return indicator_count >= 2
    
    async def _generate_solution(
        self,
        problem_text: str,
        context_data: Dict
    ) -> Dict[str, Any]:
        """
        Generate step-by-step solution using LLM.
        
        Args:
            problem_text: The problem to solve
            context_data: Retrieved NCERT context
        
        Returns:
            Dictionary with generated solution and metadata
        """
        # Build solution generation prompt
        prompt = self._build_solution_prompt(problem_text, context_data)
        
        # Generate solution with settings optimized for mathematical reasoning
        result = self.llm.generate(
            prompt=prompt,
            max_tokens=2000,  # Increased for detailed mathematical solutions
            temperature=0.1,  # Very low temperature for mathematical accuracy
            stop=["PROBLEM TO SOLVE:", "Relevant NCERT Content:"]
        )
        
        return result
    
    def _build_solution_prompt(
        self,
        problem_text: str,
        context_data: Dict
    ) -> str:
        """
        Build a well-engineered prompt for solution generation.
        
        Args:
            problem_text: The problem to solve
            context_data: Retrieved NCERT context
        
        Returns:
            Formatted prompt string
        """
        # System instruction with emphasis on mathematical accuracy
        system_prompt = """You are GuruAI, an expert mathematics tutor for JEE and NEET preparation with deep expertise in Physics, Chemistry, and Mathematics.

CRITICAL INSTRUCTIONS:
1. READ THE PROBLEM CAREFULLY - Identify all given information, constraints, and what is being asked
2. IDENTIFY THE CORRECT APPROACH - Recognize the type of problem (algebra, calculus, geometry, etc.) and select appropriate formulas
3. SHOW ALL WORK - Every step must be mathematically rigorous and logically connected
4. VERIFY YOUR ANSWER - Check if your solution makes sense given the constraints
5. USE STANDARD NOTATION - Use proper mathematical symbols and notation

SOLUTION FORMAT (MANDATORY):
Step 1: Identify given information and what needs to be found
[List all given values, constraints, and the question]

Step 2: Recall relevant formulas and concepts
[State the exact formulas needed with proper notation]

Step 3: Set up the mathematical relationships
[Show how the formulas apply to this specific problem]

Step 4: Solve step-by-step
[Show detailed calculations with clear algebraic steps]

Step 5: Verify and state the final answer
[Check if answer satisfies all constraints, state final answer clearly]

COMMON MISTAKES TO AVOID:
- Don't skip algebraic steps
- Don't confuse similar formulas (e.g., hyperbola vs ellipse)
- Don't ignore constraints or domain restrictions
- Don't make arithmetic errors
- Always check units and dimensions"""
        
        # Context section
        context = context_data.get('context', '')
        references = context_data.get('references', [])
        
        context_section = f"""Relevant NCERT Content:
{context}

References: {', '.join(references) if references else 'NCERT textbooks'}"""
        
        # Complete prompt with emphasis on accuracy
        prompt = f"""{system_prompt}

{context_section}

PROBLEM TO SOLVE:
{problem_text}

IMPORTANT: This is a JEE/NEET level problem. Take your time to:
1. Understand what type of problem this is (conic sections, mechanics, thermodynamics, etc.)
2. Identify the correct formulas and relationships
3. Apply them systematically with clear reasoning
4. Verify your answer makes mathematical sense

Now provide your detailed, mathematically rigorous solution:

Solution:"""
        
        return prompt
    
    def _parse_solution_steps(self, solution_text: str) -> List[Dict[str, str]]:
        """
        Parse solution text into individual steps.
        
        Args:
            solution_text: Complete solution text
        
        Returns:
            List of step dictionaries with step number and content
        """
        steps = []
        
        # Pattern to match steps: "Step 1:", "Step 2:", etc.
        step_pattern = r'Step\s+(\d+):\s*(.+?)(?=Step\s+\d+:|Final Answer:|$)'
        
        matches = re.finditer(step_pattern, solution_text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            step_num = match.group(1)
            step_content = match.group(2).strip()
            
            steps.append({
                'step_number': int(step_num),
                'content': step_content
            })
        
        # If no steps found, treat entire solution as one step
        if not steps:
            # Check for "Final Answer" to separate it
            if 'Final Answer' in solution_text or 'final answer' in solution_text.lower():
                parts = re.split(r'Final Answer:', solution_text, flags=re.IGNORECASE)
                if len(parts) == 2:
                    steps.append({
                        'step_number': 1,
                        'content': parts[0].strip()
                    })
            else:
                steps.append({
                    'step_number': 1,
                    'content': solution_text.strip()
                })
        
        logger.debug(f"Parsed {len(steps)} steps from solution")
        return steps
    
    def _extract_formulas(
        self,
        solution_text: str,
        context_data: Dict
    ) -> List[Dict[str, str]]:
        """
        Extract formulas from solution and add NCERT references.
        
        Args:
            solution_text: Complete solution text
            context_data: Retrieved NCERT context
        
        Returns:
            List of formula dictionaries with formula and NCERT reference
        """
        formulas = []
        
        # Common formula patterns
        # Pattern 1: F = ma, v = u + at, etc.
        formula_pattern1 = r'([A-Za-z]+\s*=\s*[A-Za-z0-9\s\+\-\*/\(\)]+)'
        
        # Pattern 2: Equations with Greek letters or special symbols
        formula_pattern2 = r'([A-Za-z]+\s*=\s*[^.!?]+)'
        
        # Find all formulas
        matches1 = re.findall(formula_pattern1, solution_text)
        
        # Get unique formulas
        unique_formulas = list(set(matches1))
        
        # Try to find NCERT references for each formula
        for formula in unique_formulas[:10]:  # Limit to 10 formulas
            # Clean up formula
            formula_clean = formula.strip()
            
            # Try to find this formula in context
            ncert_ref = self._find_formula_reference(formula_clean, context_data)
            
            formulas.append({
                'formula': formula_clean,
                'ncert_reference': ncert_ref
            })
        
        logger.debug(f"Extracted {len(formulas)} formulas")
        return formulas
    
    def _find_formula_reference(
        self,
        formula: str,
        context_data: Dict
    ) -> str:
        """
        Find NCERT reference for a formula.
        
        Args:
            formula: Formula string
            context_data: Retrieved NCERT context
        
        Returns:
            NCERT reference string or "NCERT textbook" if not found
        """
        # Check if formula appears in context
        context = context_data.get('context', '')
        
        if formula in context:
            # Try to find which reference it came from
            references = context_data.get('references', [])
            if references:
                return references[0]  # Return first reference
        
        # Default reference
        return "NCERT textbook"
    
    def _format_error_response(
        self,
        error: str,
        details: str,
        suggestion: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format error response with helpful information.
        
        Args:
            error: Error message
            details: Detailed error information
            suggestion: Suggestion for user to fix the issue
        
        Returns:
            Formatted error response dictionary
        """
        response = {
            'success': False,
            'error': error,
            'error_details': details,
            'problem_text': '',
            'solution': '',
            'steps': [],
            'formulas': [],
            'diagrams': [],
            'references': [],
            'confidence': 'low'
        }
        
        if suggestion:
            response['suggestion'] = suggestion
        
        return response
    
    def validate_image_for_problem(self, image_data: bytes) -> tuple[bool, Optional[str]]:
        """
        Validate that an image is suitable for problem solving.
        
        Args:
            image_data: Raw image bytes
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.image_processor.validate_image(image_data)
