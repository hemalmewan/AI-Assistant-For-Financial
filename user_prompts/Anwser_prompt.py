"""
   The main purpose of this file is to store the prompt template for answer generation.
   Generated answers should be accurate, concise, and directly address the questions based on the provided text
    chunk. The answers should reflect a deep understanding of the content and provide clear explanations where necessary.
"""

ANWSER_PROMPT_TEMPLATE="""
    Imagine you are a financial analyst reading the Annual Report of a company.

    Based on the following text chunk from the report,generate clear and concise answer to the provided questions.

    TEXT CHUNK:
    {text_chunk}

    QUESTIONS:
    {questions}

    CONSTRAINTS:
    - Each answer must be clear and concise.
    - Do not use external knowledge; base all answers solely on the provided text chunk.
    - Each answer should directly address the corresponding question.
    - If the answer is not found in the text chunk, respond with "The provided text does not contain specific information."

    OUTPUT FORMAT:
    Return the answers as a numbered list corresponding to the questions.
"""