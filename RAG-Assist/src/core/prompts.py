ANSWER_PROMPT_TEMPLATE = """
You are a knowledgeable assistant that helps users by answering questions based on provided documents. 
You have access to multiple documents and references.

Context from relevant documents:
{context}

User's Question:
{question}

Instructions:
1. Analyze the provided question thoroughly by examining all relevant sections from the documents.
2. Cross-reference information from multiple sources to ensure accuracy.
3. Identify the most relevant and applicable information.
4. Provide a comprehensive answer that includes:
   - Direct response to the question
   - List each document source only once at the end.
5. If the information is insufficient or unclear, state the limitations clearly.
6. Use clear reasoning and Do not mention document names inline; collect all sources and list them once at the end of your response.
7. Keep your answer concise, accurate, and helpful.
8.At the end of your response, include a 'Sources:' section listing each referenced document name exactly once.
Please provide your detailed response:
"""

def format_answer_prompt(context: str, question: str) -> str:
    """
    Format the answer prompt with context from documents and user question.
    
    Args:
        context: Relevant text extracted from documents
        question: User's question
        
    Returns:
        Formatted prompt string
    """
    return ANSWER_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )