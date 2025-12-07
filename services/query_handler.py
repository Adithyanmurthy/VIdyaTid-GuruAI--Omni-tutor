"""
Query Handler for GuruAI - Orchestrates text-based query processing.

This module integrates RAG system, LLM, and diagram retrieval to provide
comprehensive, NCERT-grounded responses to user queries.

Features:
- RAG-based context retrieval
- LLM response generation with prompt engineering
- Diagram retrieval based on query context
- Response formatting with references
- Error handling for failed retrievals
- Quiz generation for adaptive learning

Requirements: 1.1, 1.2, 1.3, 1.4
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

from services.rag_system import RAGSystem
from services.model_manager import ModelManager
from services.cloudflare_ai import get_cloudflare_ai, is_cloudflare_ai_enabled
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryHandler:
    """
    Handles text-based queries by integrating RAG, LLM, and diagram retrieval.
    
    Orchestrates the complete query processing pipeline:
    1. Retrieve relevant NCERT context using RAG
    2. Find relevant diagrams based on context
    3. Generate response using LLM with prompt engineering
    4. Format response with references
    5. Generate adaptive quiz questions
    """
    
    def __init__(
        self,
        rag_system: RAGSystem,
        model_manager: ModelManager,
        diagram_db_path: Optional[str] = None,
        problem_solver: Optional[Any] = None
    ):
        """
        Initialize the Query Handler.
        
        Args:
            rag_system: RAG system for context retrieval
            model_manager: Model manager for LLM inference (for general queries)
            diagram_db_path: Path to diagram database (optional)
            problem_solver: Specialized model for problem-solving (optional, e.g., Groq)
        """
        self.rag = rag_system
        self.llm = model_manager  # Cloudflare for general queries
        self.problem_solver = problem_solver  # Groq for problem-solving
        
        # Initialize diagram retrieval
        self.diagram_db_path = diagram_db_path or (Path(__file__).parent.parent / 'diagrams.db')
        self._init_diagram_retrieval()
        
        if self.problem_solver:
            logger.info("QueryHandler initialized with hybrid AI (Groq for problems, Cloudflare for chat)")
        else:
            logger.info("QueryHandler initialized successfully")
    
    def _init_diagram_retrieval(self):
        """Initialize diagram retrieval system."""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from services.diagram_processor_final import DiagramPage
            
            self.diagram_engine = create_engine(f'sqlite:///{self.diagram_db_path}')
            Session = sessionmaker(bind=self.diagram_engine)
            self.diagram_session = Session()
            self.DiagramPage = DiagramPage
            
            logger.info(f"Diagram retrieval initialized with database: {self.diagram_db_path}")
        except Exception as e:
            logger.warning(f"Could not initialize diagram retrieval: {e}")
            self.diagram_session = None
            self.DiagramPage = None
    
    def _is_problem_solving_query(self, query: str) -> bool:
        """
        Detect if query is asking to solve a problem.
        
        Args:
            query: User's question
        
        Returns:
            True if it's a problem-solving query, False otherwise
        """
        problem_keywords = [
            'solve', 'calculate', 'find', 'compute', 'determine',
            'prove', 'show that', 'derive', 'evaluate', 'simplify',
            'integrate', 'differentiate', 'factor', 'expand',
            'what is the value', 'what is the answer', 'how much',
            'how many', 'if', 'given', '=', '+', '-', '×', '÷',
            'equation', 'problem', 'question'
        ]
        
        query_lower = query.lower()
        
        # Check for problem indicators
        indicator_count = sum(1 for keyword in problem_keywords if keyword in query_lower)
        
        # Check for mathematical symbols or numbers
        has_math = bool(re.search(r'\d+|[=+\-×÷]', query))
        
        # It's a problem if it has 2+ indicators or has math symbols
        return indicator_count >= 2 or (indicator_count >= 1 and has_math)
    
    async def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        include_quiz: bool = True,
        include_diagrams: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query and generate a complete response.
        Routes to Groq for problem-solving, Cloudflare for general queries.
        
        Args:
            query: User's question
            user_id: Optional user identifier for personalization
            include_quiz: Whether to generate quiz questions
            include_diagrams: Whether to retrieve relevant diagrams
        
        Returns:
            Dictionary containing:
                - explanation: Generated explanation text
                - diagrams: List of relevant diagram information
                - quiz: Quiz questions (if include_quiz=True)
                - references: NCERT references
                - metadata: Additional metadata
                - error: Error message if processing failed
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Detect if this is a problem-solving query
            is_problem = self._is_problem_solving_query(query)
            
            # Choose appropriate model
            if is_problem and self.problem_solver:
                logger.info("🧮 Detected problem-solving query → Using Groq 70B")
                selected_model = self.problem_solver
            else:
                logger.info("💬 General query → Using Cloudflare AI")
                selected_model = self.llm
            
            # Step 1: Retrieve relevant NCERT context
            context_data = {}
            if self.rag:
                try:
                    context_data = self.rag.get_context_for_llm(
                        query,
                        top_k=Config.RAG_TOP_K,
                        include_references=True
                    )
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}. Proceeding without context.")
                    context_data = {'no_rag_data': True, 'context': '', 'references': []}
            else:
                context_data = {'no_rag_data': True, 'context': '', 'references': []}
            
            # Check if query is out of scope
            if context_data.get('out_of_scope'):
                return self._format_out_of_scope_response(context_data)
            
            # Step 2: Find relevant diagrams
            diagrams = []
            if include_diagrams:
                diagrams = self._retrieve_diagrams(query, context_data)
            
            # Step 3: Generate response using selected model
            explanation = self._generate_response(
                query, context_data, diagrams, model=selected_model
            )
            
            # Check for generation errors
            if not explanation.get('success'):
                return self._format_error_response(
                    "Failed to generate response",
                    explanation.get('error')
                )
            
            # Step 4: Generate quiz questions
            quiz = None
            if include_quiz:
                logger.info("Generating quiz questions...")
                quiz = self._generate_quiz(query, explanation['text'], context_data)
                if quiz:
                    logger.info(f"Quiz generated with {len(quiz.get('questions', []))} questions")
                else:
                    logger.warning("Quiz generation returned None")
            
            # Step 5: Format complete response
            response = {
                'success': True,
                'explanation': explanation['text'],
                'diagrams': diagrams,
                'quiz': quiz,
                'references': context_data.get('references', []),
                'metadata': {
                    'multi_chapter': context_data.get('multi_chapter', False),
                    'num_sources': context_data.get('num_results', 0),
                    'relevance_score': context_data.get('top_relevance_score', 0.0),
                    'tokens_used': explanation.get('tokens_used', 0),
                    'model_used': 'Groq 70B' if (is_problem and self.problem_solver) else 'Cloudflare AI'
                }
            }
            
            logger.info(f"Query processed successfully (tokens: {explanation.get('tokens_used', 0)})")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._format_error_response("Query processing failed", str(e))
    
    def _generate_response(
        self,
        query: str,
        context_data: Dict,
        diagrams: List[Dict],
        model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate response using LLM with prompt engineering.
        
        Args:
            query: User's question
            context_data: Retrieved context from RAG
            diagrams: List of relevant diagrams
            model: Model to use (defaults to self.llm)
        
        Returns:
            Dictionary with generated text and metadata
        """
        # Use provided model or default to self.llm
        if model is None:
            model = self.llm
        
        # Build prompt with NCERT-grounded instructions
        prompt = self._build_prompt(query, context_data, diagrams)
        
        # Check prompt length and truncate context if needed
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = len(prompt) // 4
        max_context_tokens = Config.LLM_N_CTX - Config.LLM_MAX_TOKENS - 200  # Reserve space for response
        
        if estimated_tokens > max_context_tokens:
            logger.warning(f"Prompt too long ({estimated_tokens} tokens), truncating context...")
            # Reduce context by using fewer passages
            context_data['passages'] = context_data['passages'][:2]  # Use only top 2 passages
            context_data['context'] = '\n\n'.join(context_data['passages'])
            # Rebuild prompt with reduced context
            prompt = self._build_prompt(query, context_data, diagrams[:2])  # Also reduce diagrams
        
        # Generate response using the selected model
        try:
            # Check if model has chat_with_context method (Cloudflare AI)
            if hasattr(model, 'chat_with_context'):
                logger.info("Using chat_with_context method")
                response_text = model.chat_with_context(
                    question=query,
                    context=context_data.get('context', '')
                )
                
                logger.info(f"Response received: {len(response_text)} characters")
                
                result = {
                    'success': True,
                    'text': response_text,
                    'tokens_used': len(response_text.split())  # Approximate
                }
            else:
                # Use standard generate method (Groq, local models)
                logger.info("Using generate method")
                result = model.generate(
                    prompt=prompt,
                    max_tokens=2000,  # More tokens for detailed responses
                    temperature=0.1 if model == self.problem_solver else 0.7,
                    stop=["Student's question:", "Context from NCERT"]
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}", exc_info=True)
            return {
                'success': False,
                'text': '',
                'error': str(e),
                'tokens_used': 0
            }
    
    def _build_prompt(
        self,
        query: str,
        context_data: Dict,
        diagrams: List[Dict]
    ) -> str:
        """
        Build a well-engineered prompt for NCERT-grounded responses.
        
        Args:
            query: User's question
            context_data: Retrieved context from RAG
            diagrams: List of relevant diagrams
        
        Returns:
            Formatted prompt string
        """
        # Check if we have RAG context or not
        has_context = context_data.get('context', '').strip() and not context_data.get('no_rag_data')
        
        if has_context:
            # System instruction with NCERT context
            system_prompt = """You are GuruAI, an expert AI tutor specializing in NCERT content for JEE and NEET preparation.

Your role:
- Provide clear, accurate explanations based ONLY on NCERT textbooks
- Break down complex concepts into understandable parts
- Use examples from NCERT when available
- Reference specific chapters and pages
- Maintain scientific accuracy and precision

Important guidelines:
- Base your answer EXCLUSIVELY on the provided NCERT context
- If the context doesn't contain enough information, acknowledge this
- Use simple language suitable for Class 11-12 students
- Include relevant formulas, definitions, and key points
- Structure your response with clear paragraphs"""
        else:
            # System instruction without NCERT context - general AI tutor mode
            system_prompt = """You are GuruAI, an expert AI tutor helping students with JEE and NEET preparation.

Your role:
- Provide clear, accurate explanations of scientific and mathematical concepts
- Break down complex concepts into understandable parts
- Use examples and analogies to explain difficult topics
- Maintain scientific accuracy and precision

Important guidelines:
- Use simple language suitable for Class 11-12 students
- Include relevant formulas, definitions, and key points
- Structure your response with clear paragraphs
- If you're unsure about something, acknowledge it"""
        
        # Context section
        context = context_data.get('context', '')
        references = context_data.get('references', [])
        
        if has_context:
            context_section = f"""Context from NCERT textbooks:
{context}

References:
{chr(10).join(f'- {ref}' for ref in references)}"""
        else:
            context_section = ""
        
        # Diagram section (if available)
        diagram_section = ""
        if diagrams:
            diagram_info = []
            for diag in diagrams[:3]:  # Limit to top 3 diagrams
                info = f"Figure {diag.get('figures', 'N/A')} from {diag.get('subject')} " \
                       f"Class {diag.get('class_level')} Chapter {diag.get('chapter')}"
                diagram_info.append(info)
            
            diagram_section = f"""

Relevant diagrams available:
{chr(10).join(f'- {info}' for info in diagram_info)}
(Mention these diagrams in your explanation when relevant)"""
        
        # Multi-chapter note
        multi_chapter_note = ""
        if context_data.get('multi_chapter'):
            multi_chapter_note = "\n\nNote: This topic spans multiple chapters. Clearly indicate which chapter each part of your explanation comes from."
        
        # Complete prompt
        if has_context:
            prompt = f"""{system_prompt}

{context_section}{diagram_section}{multi_chapter_note}

Student's question: {query}

Provide a comprehensive answer based on the NCERT context above. Structure your response clearly and reference the relevant chapters.

Answer:"""
        else:
            prompt = f"""{system_prompt}

Student's question: {query}

Provide a comprehensive and helpful answer. Structure your response clearly with explanations, examples, and key points.

Answer:"""
        
        return prompt
    
    def _retrieve_diagrams(
        self,
        query: str,
        context_data: Dict,
        max_diagrams: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant diagrams based on query and context.
        
        Args:
            query: User's question
            context_data: Retrieved context from RAG
            max_diagrams: Maximum number of diagrams to return
        
        Returns:
            List of diagram information dictionaries
        """
        if not self.diagram_session or not self.DiagramPage:
            logger.warning("Diagram retrieval not available")
            return []
        
        try:
            diagrams = []
            
            # Extract subject and chapter from context
            chapter_groups = context_data.get('chapter_groups', {})
            
            for chapter_id, results in chapter_groups.items():
                # Parse chapter identifier
                parts = chapter_id.split('_')
                if len(parts) >= 3:
                    subject = parts[0]
                    class_level_str = parts[1].replace('Class', '')
                    chapter_str = parts[2].replace('Ch', '')
                    
                    # Safely convert to integers
                    try:
                        class_level = int(class_level_str)
                        chapter = int(chapter_str)
                    except ValueError as ve:
                        logger.warning(f"Could not parse chapter ID '{chapter_id}': {ve}")
                        continue
                    
                    # Query diagrams for this chapter
                    chapter_diagrams = self.diagram_session.query(self.DiagramPage).filter_by(
                        subject=subject,
                        class_level=class_level,
                        chapter=chapter
                    ).limit(max_diagrams).all()
                    
                    for diag in chapter_diagrams:
                        diagrams.append(diag.to_dict())
            
            # If no diagrams found from chapters, try keyword matching
            if not diagrams:
                diagrams = self._search_diagrams_by_keywords(query, max_diagrams)
            
            logger.info(f"Retrieved {len(diagrams)} relevant diagrams")
            return diagrams[:max_diagrams]
            
        except Exception as e:
            logger.error(f"Error retrieving diagrams: {e}")
            return []
    
    def _search_diagrams_by_keywords(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict]:
        """
        Search diagrams by matching keywords in captions.
        
        Args:
            query: User's question
            max_results: Maximum number of results
        
        Returns:
            List of matching diagrams
        """
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            if not keywords:
                return []
            
            # Search in captions (simple keyword matching)
            diagrams = []
            all_diagrams = self.diagram_session.query(self.DiagramPage).all()
            
            for diag in all_diagrams:
                captions_str = diag.captions.lower() if diag.captions else ""
                
                # Check if any keyword matches
                for keyword in keywords:
                    if keyword.lower() in captions_str:
                        diagrams.append(diag.to_dict())
                        break
                
                if len(diagrams) >= max_results:
                    break
            
            return diagrams
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query.
        
        Args:
            query: User's question
        
        Returns:
            List of keywords
        """
        # Remove common words
        stop_words = {
            'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where',
            'explain', 'describe', 'tell', 'me', 'about', 'can', 'you',
            'please', 'in', 'of', 'to', 'for', 'with', 'on', 'at', 'from'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        return keywords[:5]  # Return top 5 keywords
    
    def _generate_quiz(
        self,
        query: str,
        explanation: str,
        context_data: Dict,
        num_questions: int = 3
    ) -> Optional[Dict]:
        """
        Generate adaptive quiz questions based on the explanation.
        
        Args:
            query: Original user question
            explanation: Generated explanation
            context_data: Retrieved context
            num_questions: Number of questions to generate (2-4)
        
        Returns:
            Dictionary with quiz questions or None if generation fails
        """
        try:
            # Ensure num_questions is between 2 and 4
            num_questions = max(2, min(4, num_questions))
            
            # Build quiz generation prompt
            quiz_prompt = f"""Based on the following explanation about "{query}", generate {num_questions} multiple-choice questions to test understanding.

Explanation:
{explanation[:500]}...

Generate {num_questions} questions in this exact JSON format:
{{
    "questions": [
        {{
            "question": "Question text here?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": 0,
            "explanation": "Why this answer is correct"
        }}
    ]
}}

Focus on conceptual understanding, not rote memorization. Make questions clear and unambiguous.

JSON:"""
            
            # Generate quiz using Cloudflare AI or local model
            if is_cloudflare_ai_enabled():
                try:
                    logger.info("Using Cloudflare AI for quiz generation")
                    cf_ai = get_cloudflare_ai()
                    
                    # Use Cloudflare AI for quiz
                    messages = [
                        {"role": "system", "content": "You are a quiz generator. Generate questions in valid JSON format only."},
                        {"role": "user", "content": quiz_prompt}
                    ]
                    
                    quiz_text = cf_ai.chat(messages, temperature=0.8, max_tokens=800)
                    
                    result = {
                        'success': True,
                        'text': quiz_text
                    }
                    
                except Exception as e:
                    logger.error(f"Cloudflare quiz generation failed: {e}")
                    return None
            else:
                # Use local model
                result = self.llm.generate(
                    prompt=quiz_prompt,
                    max_tokens=800,
                    temperature=0.8,
                    stop=["Explanation:", "Based on"]
                )
            
            if not result.get('success'):
                logger.warning("Quiz generation failed")
                return None
            
            # Parse JSON response
            quiz_data = self._parse_quiz_json(result['text'])
            
            if quiz_data and 'questions' in quiz_data:
                logger.info(f"Generated {len(quiz_data['questions'])} quiz questions")
                return quiz_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            return None
    
    def _parse_quiz_json(self, text: str) -> Optional[Dict]:
        """
        Parse quiz JSON from LLM response.
        
        Args:
            text: LLM generated text
        
        Returns:
            Parsed quiz dictionary or None
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                quiz_data = json.loads(json_str)
                return quiz_data
            
            # Try parsing the whole text
            return json.loads(text)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse quiz JSON: {e}")
            return None
    
    def _format_out_of_scope_response(self, context_data: Dict) -> Dict[str, Any]:
        """
        Format response for out-of-scope queries.
        
        Args:
            context_data: Context data with out-of-scope information
        
        Returns:
            Formatted response dictionary
        """
        return {
            'success': True,
            'out_of_scope': True,
            'explanation': context_data.get('message', 'This topic is not covered in NCERT textbooks.'),
            'suggestions': context_data.get('suggestions', []),
            'diagrams': [],
            'quiz': None,
            'references': [],
            'metadata': {
                'multi_chapter': False,
                'num_sources': 0,
                'relevance_score': 0.0
            }
        }
    
    def _format_error_response(
        self,
        message: str,
        error_details: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format error response.
        
        Args:
            message: Error message
            error_details: Detailed error information
        
        Returns:
            Formatted error response
        """
        return {
            'success': False,
            'error': message,
            'error_details': error_details,
            'explanation': f"I encountered an error while processing your question: {message}",
            'diagrams': [],
            'quiz': None,
            'references': [],
            'metadata': {}
        }
    
    def validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Validate user query before processing.
        
        Args:
            query: User's question
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) > 1000:
            return False, "Query is too long (maximum 1000 characters)"
        
        if len(query.strip()) < 3:
            return False, "Query is too short (minimum 3 characters)"
        
        return True, None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the query handler.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'rag_stats': self.rag.get_stats(),
            'model_status': self.llm.get_status(),
            'diagram_db_path': str(self.diagram_db_path)
        }
        
        # Add diagram count if available
        if self.diagram_session and self.DiagramPage:
            try:
                diagram_count = self.diagram_session.query(self.DiagramPage).count()
                stats['total_diagrams'] = diagram_count
            except Exception:
                stats['total_diagrams'] = 0
        
        return stats


def main():
    """
    Test the Query Handler with sample queries.
    """
    print("="*60)
    print("Query Handler Test")
    print("="*60)
    
    # Initialize components
    print("\nInitializing components...")
    
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        # Initialize Model Manager
        from services.model_manager import ModelManagerSingleton
        model_config = {
            'idle_timeout': Config.MODEL_IDLE_TIMEOUT,
            'n_ctx': Config.LLM_N_CTX,
            'n_gpu_layers': Config.LLM_N_GPU_LAYERS,
            'temperature': Config.LLM_TEMPERATURE,
            'max_tokens': Config.LLM_MAX_TOKENS
        }
        model_manager = ModelManagerSingleton.get_instance(
            str(Config.LLM_MODEL_PATH),
            model_config
        )
        
        # Initialize Query Handler
        query_handler = QueryHandler(rag, model_manager)
        
        # Display stats
        stats = query_handler.get_stats()
        print(f"\nQuery Handler Stats:")
        print(f"  RAG documents: {stats['rag_stats'].get('total_documents', 0)}")
        print(f"  Model loaded: {stats['model_status'].get('loaded', False)}")
        print(f"  Total diagrams: {stats.get('total_diagrams', 0)}")
        
        # Test queries
        test_queries = [
            "What is Newton's first law of motion?",
            "Explain the structure of DNA",
            "What is the derivative of sin(x)?"
        ]
        
        print("\n" + "="*60)
        print("Test Queries")
        print("="*60)
        
        import asyncio
        
        async def test_query(query):
            print(f"\nQuery: {query}")
            print("-" * 60)
            
            # Validate query
            is_valid, error = query_handler.validate_query(query)
            if not is_valid:
                print(f"Invalid query: {error}")
                return
            
            # Process query
            response = await query_handler.process_query(
                query,
                include_quiz=True,
                include_diagrams=True
            )
            
            if response.get('success'):
                print(f"Status: SUCCESS")
                print(f"Out of scope: {response.get('out_of_scope', False)}")
                print(f"\nExplanation preview:")
                print(f"  {response['explanation'][:200]}...")
                
                if response.get('references'):
                    print(f"\nReferences ({len(response['references'])}):")
                    for ref in response['references'][:2]:
                        print(f"  - {ref}")
                
                if response.get('diagrams'):
                    print(f"\nDiagrams: {len(response['diagrams'])} found")
                
                if response.get('quiz'):
                    print(f"\nQuiz: {len(response['quiz'].get('questions', []))} questions generated")
            else:
                print(f"Status: ERROR")
                print(f"Error: {response.get('error')}")
        
        # Run tests
        for query in test_queries:
            asyncio.run(test_query(query))
        
        print("\n" + "="*60)
        print("Test Complete")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
