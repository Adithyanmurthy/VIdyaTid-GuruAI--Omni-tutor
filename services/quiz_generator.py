"""
Quiz Generator for GuruAI - Generates adaptive quiz questions based on explanations.

This module creates multiple-choice questions (MCQs) to test student understanding
of explained concepts. It generates 2-4 questions per explanation with proper
answer validation and feedback.

Features:
- MCQ generation with 4 options
- 2-4 questions per explanation
- Answer validation
- Detailed feedback for incorrect answers
- Conceptual understanding focus (not rote memorization)

Requirements: 10.1, 10.3, 10.4
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from services.model_manager import ModelManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QuizQuestion:
    """Represents a single quiz question."""
    question: str
    options: List[str]  # Exactly 4 options
    correct_answer: int  # Index of correct option (0-3)
    explanation: str  # Explanation for the correct answer
    incorrect_explanations: Dict[int, str]  # Explanations for why wrong answers are wrong
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class QuizGenerator:
    """
    Generates adaptive quiz questions based on explained concepts.
    
    Creates MCQ questions that test conceptual understanding rather than
    rote memorization. Provides immediate feedback and explanations.
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize the Quiz Generator.
        
        Args:
            model_manager: Model manager for LLM inference
        """
        self.llm = model_manager
        logger.info("QuizGenerator initialized successfully")
    
    def generate_quiz(
        self,
        query: str,
        explanation: str,
        context: Optional[str] = None,
        num_questions: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate adaptive quiz questions based on the explanation.
        
        Args:
            query: Original user question
            explanation: Generated explanation text
            context: Optional NCERT context
            num_questions: Number of questions to generate (2-4, default: 3)
        
        Returns:
            Dictionary containing:
                - success: Whether generation succeeded
                - questions: List of QuizQuestion objects
                - num_questions: Actual number of questions generated
                - error: Error message if generation failed
        """
        try:
            # Validate and set number of questions (2-4 range)
            if num_questions is None:
                num_questions = 3
            num_questions = max(2, min(4, num_questions))
            
            logger.info(f"Generating {num_questions} quiz questions for query: {query[:50]}...")
            
            # Build quiz generation prompt
            prompt = self._build_quiz_prompt(query, explanation, context, num_questions)
            
            # Generate quiz using LLM
            result = self.llm.generate(
                prompt=prompt,
                max_tokens=1200,
                temperature=0.8,
                stop=["###", "Explanation:", "Based on the following"]
            )
            
            if not result.get('success'):
                logger.error("LLM generation failed for quiz")
                return {
                    'success': False,
                    'error': 'Failed to generate quiz questions',
                    'questions': [],
                    'num_questions': 0
                }
            
            # Parse the generated quiz
            questions = self._parse_quiz_response(result['text'], num_questions)
            
            if not questions:
                logger.warning("No valid questions parsed from LLM response")
                return {
                    'success': False,
                    'error': 'Failed to parse quiz questions',
                    'questions': [],
                    'num_questions': 0
                }
            
            logger.info(f"Successfully generated {len(questions)} quiz questions")
            
            return {
                'success': True,
                'questions': [q.to_dict() for q in questions],
                'num_questions': len(questions),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error generating quiz: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'questions': [],
                'num_questions': 0
            }
    
    def _build_quiz_prompt(
        self,
        query: str,
        explanation: str,
        context: Optional[str],
        num_questions: int
    ) -> str:
        """
        Build a prompt for quiz generation.
        
        Args:
            query: Original user question
            explanation: Generated explanation
            context: Optional NCERT context
            num_questions: Number of questions to generate
        
        Returns:
            Formatted prompt string
        """
        # Truncate explanation if too long
        max_explanation_length = 800
        if len(explanation) > max_explanation_length:
            explanation = explanation[:max_explanation_length] + "..."
        
        prompt = f"""You are an expert educator creating quiz questions to test student understanding of NCERT concepts.

Topic: {query}

Explanation provided to student:
{explanation}

Generate exactly {num_questions} multiple-choice questions that test conceptual understanding (not rote memorization).

Requirements for each question:
1. Question should test understanding, not just recall
2. Provide exactly 4 options (A, B, C, D)
3. Only one option should be correct
4. Include explanation for the correct answer
5. Include brief explanations for why each incorrect option is wrong

Format your response as JSON:
{{
    "questions": [
        {{
            "question": "Clear question text here?",
            "options": [
                "Option A text",
                "Option B text",
                "Option C text",
                "Option D text"
            ],
            "correct_answer": 0,
            "explanation": "Why option A is correct and relates to the concept",
            "incorrect_explanations": {{
                "1": "Why option B is incorrect",
                "2": "Why option C is incorrect",
                "3": "Why option D is incorrect"
            }}
        }}
    ]
}}

Generate {num_questions} questions now:
"""
        
        return prompt
    
    def _parse_quiz_response(
        self,
        response_text: str,
        expected_count: int
    ) -> List[QuizQuestion]:
        """
        Parse quiz questions from LLM response.
        
        Args:
            response_text: Raw text from LLM
            expected_count: Expected number of questions
        
        Returns:
            List of QuizQuestion objects
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in quiz response")
                return []
            
            json_str = json_match.group(0)
            quiz_data = json.loads(json_str)
            
            if 'questions' not in quiz_data:
                logger.warning("No 'questions' key in parsed JSON")
                return []
            
            questions = []
            for q_data in quiz_data['questions']:
                question = self._parse_single_question(q_data)
                if question:
                    questions.append(question)
            
            # Validate we got the expected number of questions
            if len(questions) < expected_count:
                logger.warning(f"Generated {len(questions)} questions, expected {expected_count}")
            
            return questions[:expected_count]  # Return at most expected_count
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse quiz JSON: {e}")
            # Try fallback parsing
            return self._fallback_parse(response_text, expected_count)
        except Exception as e:
            logger.error(f"Error parsing quiz response: {e}")
            return []
    
    def _parse_single_question(self, q_data: Dict) -> Optional[QuizQuestion]:
        """
        Parse a single question from JSON data.
        
        Args:
            q_data: Dictionary with question data
        
        Returns:
            QuizQuestion object or None if invalid
        """
        try:
            # Validate required fields
            if 'question' not in q_data or 'options' not in q_data or 'correct_answer' not in q_data:
                logger.warning("Missing required fields in question data")
                return None
            
            # Validate options count
            options = q_data['options']
            if not isinstance(options, list) or len(options) != 4:
                logger.warning(f"Invalid options count: {len(options) if isinstance(options, list) else 'not a list'}")
                return None
            
            # Validate correct answer index
            correct_answer = q_data['correct_answer']
            if not isinstance(correct_answer, int) or correct_answer < 0 or correct_answer > 3:
                logger.warning(f"Invalid correct_answer index: {correct_answer}")
                return None
            
            # Get explanations
            explanation = q_data.get('explanation', 'This is the correct answer based on NCERT concepts.')
            
            # Parse incorrect explanations
            incorrect_explanations = {}
            if 'incorrect_explanations' in q_data:
                inc_exp = q_data['incorrect_explanations']
                if isinstance(inc_exp, dict):
                    # Convert string keys to integers
                    for key, value in inc_exp.items():
                        try:
                            idx = int(key)
                            if idx != correct_answer and 0 <= idx <= 3:
                                incorrect_explanations[idx] = value
                        except (ValueError, TypeError):
                            continue
            
            # Fill in missing incorrect explanations
            for i in range(4):
                if i != correct_answer and i not in incorrect_explanations:
                    incorrect_explanations[i] = "This option is incorrect. Please review the concept."
            
            return QuizQuestion(
                question=q_data['question'],
                options=options,
                correct_answer=correct_answer,
                explanation=explanation,
                incorrect_explanations=incorrect_explanations
            )
            
        except Exception as e:
            logger.error(f"Error parsing single question: {e}")
            return None
    
    def _fallback_parse(
        self,
        response_text: str,
        expected_count: int
    ) -> List[QuizQuestion]:
        """
        Fallback parsing when JSON parsing fails.
        
        Attempts to extract questions using pattern matching.
        
        Args:
            response_text: Raw text from LLM
            expected_count: Expected number of questions
        
        Returns:
            List of QuizQuestion objects
        """
        logger.info("Attempting fallback parsing...")
        questions = []
        
        # Try to find question patterns
        # Pattern: Question: ... Options: A) ... B) ... C) ... D) ... Answer: ...
        question_pattern = r'Question:\s*(.+?)\s*Options?:\s*A\)\s*(.+?)\s*B\)\s*(.+?)\s*C\)\s*(.+?)\s*D\)\s*(.+?)\s*(?:Correct\s*)?Answer:\s*([A-D])'
        
        matches = re.finditer(question_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            try:
                question_text = match.group(1).strip()
                option_a = match.group(2).strip()
                option_b = match.group(3).strip()
                option_c = match.group(4).strip()
                option_d = match.group(5).strip()
                correct_letter = match.group(6).strip().upper()
                
                # Convert letter to index
                correct_index = ord(correct_letter) - ord('A')
                if correct_index < 0 or correct_index > 3:
                    continue
                
                # Create question
                question = QuizQuestion(
                    question=question_text,
                    options=[option_a, option_b, option_c, option_d],
                    correct_answer=correct_index,
                    explanation="This is the correct answer based on the explanation provided.",
                    incorrect_explanations={
                        i: "This option is incorrect." for i in range(4) if i != correct_index
                    }
                )
                
                questions.append(question)
                
                if len(questions) >= expected_count:
                    break
                    
            except Exception as e:
                logger.warning(f"Error in fallback parsing: {e}")
                continue
        
        logger.info(f"Fallback parsing extracted {len(questions)} questions")
        return questions
    
    def validate_answer(
        self,
        question: Dict[str, Any],
        user_answer: int
    ) -> Dict[str, Any]:
        """
        Validate a user's answer to a quiz question.
        
        Args:
            question: Question dictionary (from QuizQuestion.to_dict())
            user_answer: Index of user's selected answer (0-3)
        
        Returns:
            Dictionary containing:
                - correct: Whether the answer is correct
                - user_answer: User's answer index
                - correct_answer: Correct answer index
                - feedback: Explanation text
        """
        try:
            correct_answer = question['correct_answer']
            is_correct = (user_answer == correct_answer)
            
            if is_correct:
                feedback = question.get('explanation', 'Correct!')
            else:
                # Get explanation for the incorrect answer
                incorrect_explanations = question.get('incorrect_explanations', {})
                feedback = incorrect_explanations.get(
                    str(user_answer),
                    incorrect_explanations.get(
                        user_answer,
                        "This answer is incorrect. "
                    )
                )
                
                # Add correct answer info
                correct_option = question['options'][correct_answer]
                feedback += f"\n\nThe correct answer is: {correct_option}\n"
                feedback += question.get('explanation', '')
            
            return {
                'correct': is_correct,
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'feedback': feedback
            }
            
        except Exception as e:
            logger.error(f"Error validating answer: {e}")
            return {
                'correct': False,
                'user_answer': user_answer,
                'correct_answer': question.get('correct_answer', 0),
                'feedback': 'Error validating answer. Please try again.'
            }
    
    def get_quiz_summary(
        self,
        questions: List[Dict[str, Any]],
        user_answers: List[int]
    ) -> Dict[str, Any]:
        """
        Generate a summary of quiz performance.
        
        Args:
            questions: List of question dictionaries
            user_answers: List of user's answers (indices)
        
        Returns:
            Dictionary with performance summary:
                - total_questions: Total number of questions
                - correct_answers: Number of correct answers
                - accuracy: Percentage accuracy
                - results: List of per-question results
        """
        if len(questions) != len(user_answers):
            logger.warning("Mismatch between questions and answers count")
            return {
                'error': 'Invalid input: questions and answers count mismatch'
            }
        
        results = []
        correct_count = 0
        
        for i, (question, user_answer) in enumerate(zip(questions, user_answers)):
            validation = self.validate_answer(question, user_answer)
            results.append({
                'question_number': i + 1,
                'question': question['question'],
                'user_answer': user_answer,
                'correct_answer': question['correct_answer'],
                'is_correct': validation['correct'],
                'feedback': validation['feedback']
            })
            
            if validation['correct']:
                correct_count += 1
        
        accuracy = (correct_count / len(questions)) * 100 if questions else 0
        
        return {
            'total_questions': len(questions),
            'correct_answers': correct_count,
            'incorrect_answers': len(questions) - correct_count,
            'accuracy': round(accuracy, 2),
            'results': results
        }


def main():
    """
    Test the Quiz Generator with sample data.
    """
    print("="*60)
    print("Quiz Generator Test")
    print("="*60)
    
    # Sample explanation
    sample_query = "What is Newton's First Law of Motion?"
    sample_explanation = """
    Newton's First Law of Motion, also known as the Law of Inertia, states that:
    
    "An object at rest stays at rest and an object in motion stays in motion with 
    the same speed and in the same direction unless acted upon by an unbalanced force."
    
    This means that objects naturally resist changes in their state of motion. 
    This property is called inertia. For example, when a bus suddenly stops, 
    passengers jerk forward because their bodies want to continue moving forward 
    due to inertia.
    
    The law explains why we need to apply force to start moving an object or to 
    stop a moving object. Without external forces, objects maintain their current 
    state of motion.
    """
    
    try:
        # Initialize Model Manager
        from config import Config
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
        
        # Initialize Quiz Generator
        quiz_gen = QuizGenerator(model_manager)
        
        print("\nGenerating quiz questions...")
        print(f"Query: {sample_query}")
        print(f"Explanation length: {len(sample_explanation)} characters")
        
        # Generate quiz
        result = quiz_gen.generate_quiz(
            query=sample_query,
            explanation=sample_explanation,
            num_questions=3
        )
        
        if result['success']:
            print(f"\n✓ Successfully generated {result['num_questions']} questions")
            
            # Display questions
            for i, q in enumerate(result['questions'], 1):
                print(f"\n{'='*60}")
                print(f"Question {i}: {q['question']}")
                print(f"\nOptions:")
                for j, option in enumerate(q['options']):
                    marker = "✓" if j == q['correct_answer'] else " "
                    print(f"  {marker} {chr(65+j)}) {option}")
                
                print(f"\nCorrect Answer: {chr(65 + q['correct_answer'])}")
                print(f"Explanation: {q['explanation']}")
            
            # Test answer validation
            print(f"\n{'='*60}")
            print("Testing Answer Validation")
            print(f"{'='*60}")
            
            test_question = result['questions'][0]
            correct_idx = test_question['correct_answer']
            wrong_idx = (correct_idx + 1) % 4
            
            # Test correct answer
            print(f"\nValidating correct answer (option {chr(65 + correct_idx)}):")
            validation = quiz_gen.validate_answer(test_question, correct_idx)
            print(f"  Result: {'✓ Correct' if validation['correct'] else '✗ Incorrect'}")
            print(f"  Feedback: {validation['feedback'][:100]}...")
            
            # Test wrong answer
            print(f"\nValidating wrong answer (option {chr(65 + wrong_idx)}):")
            validation = quiz_gen.validate_answer(test_question, wrong_idx)
            print(f"  Result: {'✓ Correct' if validation['correct'] else '✗ Incorrect'}")
            print(f"  Feedback: {validation['feedback'][:100]}...")
            
            # Test quiz summary
            print(f"\n{'='*60}")
            print("Testing Quiz Summary")
            print(f"{'='*60}")
            
            # Simulate user answers (mix of correct and incorrect)
            user_answers = [
                result['questions'][0]['correct_answer'],  # Correct
                (result['questions'][1]['correct_answer'] + 1) % 4,  # Wrong
                result['questions'][2]['correct_answer']  # Correct
            ]
            
            summary = quiz_gen.get_quiz_summary(result['questions'], user_answers)
            print(f"\nQuiz Performance:")
            print(f"  Total Questions: {summary['total_questions']}")
            print(f"  Correct Answers: {summary['correct_answers']}")
            print(f"  Incorrect Answers: {summary['incorrect_answers']}")
            print(f"  Accuracy: {summary['accuracy']}%")
            
        else:
            print(f"\n✗ Quiz generation failed: {result['error']}")
        
        print(f"\n{'='*60}")
        print("Test Complete")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
