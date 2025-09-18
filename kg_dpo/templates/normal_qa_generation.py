# pylint: disable=C0301

#######################################################S2S2S2S2S2S2S2S2S2S2S2S2S2S2S2S2S2S2S2S2S2S2S2#################################################################################
TEMPLATE_CN: str = """# 任务
你作为一个专业的问答对生成器, 根据提供的知识图谱生成{number}个高质量的问答对(Q&A Pair)

# 输入
知识图谱信息：
--实体--
{entities}
#########
--关系--
{relationships}

# 输出要求
1.格式要求:请严格按照以下json格式输出，避免输出与答案无关的内容:
    {{问题1: 答案1, 问题2: 答案2, 问题3: 答案3, 问题4: 答案4, 问题5: 答案5, ...}}
2.问题(question)要求:
    核心信息聚焦：问题必须紧密围绕知识图谱信息（关键事实、重要结论、主要观点、关键数据、核心定义、关键细节等）。
    多样性：问题类型要多样化。可以包括(不仅限于):
        细节型: X是什么?Y的关键步骤有哪些?
        原因/影响型: 为什么会产生Z? W有什么影响?
        方法/过程型: 如何实现A?描述一下B的过程。
        定义/比较型: 定义C。D和E有什么区别?
        目标/用途型: F的主要目标是什么? G用于什么?
    清晰简洁: 问题表达清晰、无歧义，句式完整。
    基于知识图谱: 问题必须**完全且只能**基于所提供的知识图谱内容，**严禁**超出知识图谱范围提问或引入外部知识。
3.答案(answer)要求:
    精准对应：答案必须直接、精准、完整地回答所提出的问题。
    来源知识图谱: 答案必须严格基于知识图谱中的信息。尽可能使用知识图谱中的词汇和表述。仅当对原句进行极简微调（如提炼总结关键短语）能使答案更精炼时方可使用，但核心信息必须忠实原文。
    完整自洽: 答案本身需要是一个完整、能独立回答问题的陈述，避免仅指向其他部分。
    简洁无冗余: 避免在答案中重复问题本身，直接提供关键信息即可。

# 总体注意事项
忠实性: 问答必须100%忠实于原始知识图谱。不能添加、臆测或歪曲知识图谱内容。
唯一焦点: 一个问答对应知识图谱中的一个核心信息点。
语言质量: 问题和答案都需要语法正确、表达清晰。
# """

TEMPLATE_EN: str = """# Task
You are a professional Q&A pair generator. Generate {number} high-quality Q&A pairs based on the provided knowledge graph.

# Input
knowledge graph information:
--entities--
{entities}
#########
--relationships--
{relationships}

# Output Requirements
1.Format requirement: Strictly follow the JSON format below, avoid outputting content unrelated to the answer:
    {{Question 1: Answer 1, Question 2: Answer 2, Question 3: Answer 3, Question 4: Answer 4, Question 5: Answer 5, ...}}
2.Question requirements:
    Core information focus: Questions must closely align with the knowledge graph information (key facts, important conclusions, main viewpoints, critical data, core definitions, key details, etc.).
    Diversity: Diversify question types. Include (but are not limited to):
        Detail-oriented: What is X? What are the key steps of Y?
        Cause/effect: Why does Z occur? What is the impact of W?
        Method/process: How to achieve A? Describe the process of B.
        Definition/comparison: Define C. What is the difference between D and E?
        Purpose/application: What is the primary objective of F? What is G used for?
        Clarity and conciseness: Express questions clearly and unambiguously with complete sentence structures.
    knowledge graph-based: Questions must exclusively and solely be based on the provided knowledge graph content. Strictly prohibited: Asking questions beyond the knowledge graph scope or introducing external knowledge.
3.Answer requirements:
    Precise correspondence: Answers must directly, accurately, and comprehensively address the question.
    Source fidelity: Answers must be strictly based on information from the knowledge graph. Use vocabulary and expressions from the knowledge graph whenever possible. Minimal adjustments (e.g., refining key phrases for conciseness) are permitted only if they enhance precision, but core information must remain faithful to the original text.
    Complete self-containment: Answers must be complete, independent statements that sufficiently address the question. Avoid referencing other sections.
    Conciseness: Avoid restating the question within the answer. Provide key information directly.

# General Notes
Fidelity: Q&A pairs must be 100 percent faithful to the original knowledge graph. Do not add, speculate, or distort content.
Single focus: Each Q&A pair should address one core piece of information from the knowledge graph.
Language quality: Ensure grammatical correctness and clarity in both questions and answers.
"""

NORMAL_QA_GENERATION_PROMPT = {
    "English": TEMPLATE_EN,
    "Chinese": TEMPLATE_CN,
}