# Language: Python
# Description: DPO generate template

DPO_TEMPLATE_CN = """请为以下问题生成{negative_example_number}个具有迷惑性、增加回答难度但事实错误的答案选项。这些选项应该看起来合理但实际上是错误的。
对于生成的答案,即使是最先进的GPT-4o、Claude-4、Gemini等模型,也难以区分;你自己也难以区分。
生成答案时，您必须引用相关上下文信息，并根据上下文生成不正确的答案。

--问题--
{question}
--正确答案--
{answer}
--上下文--
{context}
请返回一个包含JSON格式错误答案的列表，如下例所示：
["错误答案1", …, "错误答案{negative_example_number}"]

要求：
1、迷惑性与准确性:每个选项需与问题紧密相关，但内容事实错误,但是不能答非所问；避免明显错误（如逻辑矛盾或常识偏差），确保长度、风格、领域、主题等与正确答案高度一致。
2、唯一性与合规性:所有选项互不重复，且与正确答案完全不同；禁止添加任何备注、说明或额外文本。
3、上下文信息绑定, 必须从上下文信息中提取具体元素（如关键事实、数字、时间、因果关系或专业术语），并通过细微扭曲生成错误点。扭曲方式不限于以下操作：
    - 替代正确的信息：用错误但相似的信息替换上下文中的具体事实（例如：将上下文中的“2023年”改为“2022年”，或将“光合作用”改为“呼吸作用”）。
    - 删除正确的信息：省略上下文中的关键限定词或细节，导致过度泛化或隐蔽错误（例如：上下文描述“在严格控制的实验中，X导致Y”，删除条件改为“X导致Y”）。
    - 增加错误信息：添加上下文未提及但看似合理的错误细节（例如：上下文讨论“量子计算”，添加“依赖经典二进制逻辑”）。
    - 时间/数字错位：修改上下文中的具体年份、数量或比例（如上下文提到“150个”，改为“151个”或“149个”）。
    - 因果颠倒：反转上下文中的逻辑关系（如上下文描述“A导致B”，改为“B导致A”或“A导致C”）。
    - 术语误用：替换上下文中的专业术语为相似但错误的词汇（如上下文用“光合作用”，改为“呼吸作用”）。
    - 逻辑矛盾：引入与上下文矛盾的逻辑（如上下文描述“A是B”，引入“A不是C”）。
    - 细节缺失：省略上下文中的重要细节（如上下文描述“人类是动物”，省略“动物”）。
4、难度控制:错误需隐蔽（例如：细微事实扭曲、时间/数字错位、专业术语误用）,使人类和AI均需深度推理才能识别
5、输出语言:你需要按照问题的语言输出答案,如果问题是中文,则答案也需要是中文;如果问题是英文,则答案也必需是英文。
限制：
直接返回列表，不需要嵌套字典，不需要任何说明和解释性文字，也不需要用``` ```包裹。
"""

DPO_TEMPLATE_EN = """Please generate {negative_example_number} deceptive answer options that increase the difficulty of answering but are factually incorrect. These options should appear plausible but actually wrong.
For the generated answers, even the most advanced models like GPT-4o, Claude-4, Gemini, etc., will struggle to distinguish them; you yourself will also find it difficult to distinguish them.
When generating answers, you must reference the relative context information and generate incorrect answers based on the context if have.

--Question--
{question}
--Correct Answer--
{answer}
--Context--
{context}

Please return a list containing incorrect answers in JSON format as shown in the following example:
["Wrong Answer 1", ..., "Wrong Answer {negative_example_number}"]

Requirements:
1. Deceptiveness and Accuracy: Each option must be closely related to the question but factually incorrect, and must not be irrelevant; avoid obvious errors (such as logical contradictions or common-sense deviations), ensuring consistency with the correct answer in length, style, domain, theme, etc.
2. Uniqueness and Compliance: All options must be mutually non-repetitive and completely different from the correct answer; adding any notes, explanations, or additional text is prohibited.
3. Contextual Information Binding: Must extract specific elements from the contextual information (such as key facts, numbers, time, causal relationships, or professional terminology), and generate error points through subtle distortion. Distortion methods include but are not limited to the following operations:
    - Replace correct information: Substitute correct facts in the context with incorrect but similar information (e.g., change "2023" to "2022" in the context, or change "photosynthesis" to "respiration").
    - Delete correct information: Omit key qualifiers or details in the context, causing over-generalization or concealed errors (e.g., context describes "in strictly controlled experiments, X causes Y"; delete the condition to become "X causes Y").
    - Add incorrect information: Add plausible but incorrect details not mentioned in the context (e.g., context discusses "quantum computing"; add "relies on classical binary logic").
    - Time/number misalignment: Modify specific years, quantities, or proportions in the context (e.g., context mentions "150," change to "151" or "149").
    - Reverse causality: Invert logical relationships in the context (e.g., context describes "A causes B," change to "B causes A" or "A causes C").
    - Terminology misuse: Replace professional terms in the context with similar but incorrect vocabulary (e.g., context uses "photosynthesis," change to "respiration").
    - Logical contradiction: Introduce logic conflicting with the context (e.g., context describes "A is B," introduce "A is not C").
    - Detail omission: Omit important details in the context (e.g., context describes "humans are animals," omit "animals").
    - Difficulty control: Errors must be concealed (e.g., subtle factual distortions, time/number misalignments, terminology misuse), requiring deep reasoning by both humans and AI to identify.
Output language: You must output answers in the language of the question; if the question is in Chinese, answers must also be in Chinese; if the question is in English, answers must be in English.

Restrictions:
Return only the list directly; no nested dictionaries, no explanatory notes or additional text of any kind, and no wrapping with ``` ```.
"""


DPO_GENERATION_PROMPT = {
    "English": DPO_TEMPLATE_EN,
    "Chinese": DPO_TEMPLATE_CN,
}