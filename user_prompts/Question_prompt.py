"""
    The main purpose of this file is to store the prompt template for question generation.
    Generated questions should be clear, concise, and relevant to the provided text chunk.
    The questions should cover key concepts and details from the text to facilitate understanding and retention. 
"""

QUESTION_PROMPT_TEMPLATE="""
  Imagine you are financial analyst an reading the Annual Report of a company.

  Baesed on the following text chunk from the report,generate 10 relevant and insighful questions 
  that can be answered using only the information provided in the text.

  TEXT CHUNK:
  {text_chunk}

  QUESTION REQUIREMENTS:
    - Generate exactly 10 questions.
    - 4 questions about hard factual information (numbers, entities, dates).
    - 4 questions about strategy, risks, or business implications.
    - 2 questions that require stylistic or creative summarization.

 CONSTRAINTS:
    -Each question must be clear and concise.
    -Do not use external knowledge; base all questions solely on the provided text chunk.
    -Do not repeat questions; each must be unique.

 OUTPUT FORMAT:
    Return the questions as a numbered list from 1 to 10.

"""