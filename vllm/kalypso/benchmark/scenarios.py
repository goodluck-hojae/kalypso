
# FEVER
FEVER_FILTER = "Given a claim and evidence, decide if the evidence is enough to determine whether the claim is true or false."
FEVER_MAP = "Explain how the evidence supports or unsupports the claim. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters."
FEVER_FACTOOL_QUERY_MAP = (
    "Generate two concise Wikipedia search queries that would retrieve evidence "
    "needed to verify the claim.\n\n"
    "Example 1\n"
    "Claim: Graham Neubig is a professor at MIT.\n"
    "Output:\n"
    "1. Graham Neubig current position\n"
    "2. Is Graham Neubig a professor at MIT?\n\n"
    "Example 2\n"
    "Claim: The Eiffel Tower is located in Berlin.\n"
    "Output:\n"
    "1. Eiffel Tower location\n"
    "2. Is the Eiffel Tower in Berlin?\n\n"
    "Example 3\n"
    "Claim: The film Titanic was directed by James Cameron.\n"
    "Output:\n"
    "1. Titanic film director\n"
    "2. James Cameron directed Titanic\n\n"
    "Return exactly two numbered search queries and no other text."
)
FEVER_FACTOOL_SUPPORT_FILTER = (
    "Based on the retrieved Wikipedia content, is the claim supported? "
    "Answer true only if the Wikipedia content clearly supports the claim."
)


# ENRON 1
FILTER_ENRON_FRAUD = (
    'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS = (
    'The email is not quoting from a news article or an article written by someone outside of Enron.'
)

MAP_ENRON_EXPLANATION = (
    'Explain briefly why this email is related to a fraudulent scheme, using the email contents provided in the context. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.'
)


FILTER_ENRON_FRAUD_2 = (
    'The email is normal scheme (i.e., Not "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'

)
FILTER_ENRON_NOT_NEWS_2 = (
    'The email includes people associated with Enron.'
)

MAP_ENRON_EXPLANATION_2 = (
    'Summarize the email briefly. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.'
)
# ARXIV

# TOPK - MAP
CASE_1_TOPK_ARXIV = ("Provide the most relevant papers to Image Semantic Segmentation research")
CASE_1_MAP_ARXIV = (
    'Explain why is the paper relevant to Image Semantic Segmentation research. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.'
    )
CASE_1_AGG_ARXIV = (
    '"Summarize the key common ideas and contributions across the paper abstracts"'
    )

# FILTER - JOIN - FILTER - MAP -
CASE_2_FILTER_ARXIV = ("Is the paper relevant to AI robotic research?")
CASE_2_JOIN_ARXIV = (
    'Do these two papers study similar topic?'
)
CASE_2_MAP_ARXIV = (
    'Explain their relationship in terms of shared topics, methods, and differences. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.'
)

CASE_3_FILTER_1 = ("Is this abstract about AI or machine learning?")
CASE_3_FILTER_2 = ("Is this abstract describing theoretical work?")
CASE_3_FILTER_3 = ("Does this abstract mention using an image dataset?")
CASE_3_MAP_ARXIV = ("Summarize the abstract in 1–2 sentences, focusing on the main method and key result. Do not add filler, repeated punctuation, or repeated characters.")



RESUME_CASE_1_FILTER = ("Does this resume show software development or programming experience?")
RESUME_CASE_1_MAP_1 = ("Summarize the resume. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.")
RESUME_CASE_1_JOIN = ("Is this candidate a plausible fit for this software/developer job based on skills, tools, and experience?")
RESUME_CASE_1_MAP = ("Explain why this candidate is suitable for the position briefly. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.")

RESUME_CASE_2_TOPK = ("I am finding a candidate who knows C++ and Machine Learning or Statistical Analysis.")
RESUME_CASE_2_FILTER = ("Does this candidate have more than 5 years of experience?")
RESUME_CASE_2_MAP = ("Can you extract the technical skills of the candidate? Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.")


RESUME_CASE_3_MAP = ("Summarize the resume. Return one concise sentence. Do not add filler, repeated punctuation, or repeated characters.")
RESUME_CASE_3_FILTER = ("Does this resume show software development or programming experience?")


# BioDEX reaction classification
BIODEX_MAP_REACTIONS = (
    "Extract adverse drug reaction terms that are explicitly described for the patient in this article. "
    "Always write your answer as a list of 2-10 comma-separated reaction labels."
)
BIODEX_JOIN_REACTION = (
    "Does the biomedical article describe a patient who experienced this adverse drug reaction?"
)


# Contract-NLI
CONTRACT_NLI_VALID_CONTRACT = (
    "Is this document a valid contract or agreement text with enough substantive "
    "clauses to evaluate confidentiality obligations?"
)
CONTRACT_NLI_ENTAILMENT_JOIN = (
    "Given the contract and hypothesis, does the contract entail the hypothesis? "
    "Answer true only if the contract clearly supports the hypothesis."
)
CONTRACT_NLI_EXPLAIN_ENTAILMENT = (
    "Explain briefly why the contract entails the hypothesis. Cite the relevant "
    "obligation or clause in one concise sentence. Do not add filler, repeated "
    "punctuation, or repeated characters."
)
CONTRACT_NLI_CATEGORY_JOIN = (
    "Does this contract-hypothesis entailment and its explanation concern the "
    "given confidentiality-obligation category? Answer true or false."
)
CONTRACT_NLI_TOPK_EVIDENCE = (
    "Which document presents stronger and more explicit contractual evidence "
    "for its stated confidentiality obligation? Answer Document A or Document B."
)
CONTRACT_NLI_SUMMARIZE_TOP_EVIDENCE = (
    "Summarize this selected confidentiality obligation and its supporting "
    "contract language in one concise sentence."
)


# MEDEC medical error detection and correction
MEDEC_ERROR_FILTER = (
    "This is the MEDEC benchmark for medical error detection in clinical notes. "
    "The note may contain one subtle deliberately injected medical error in one numbered sentence, "
    "such as an incorrect diagnosis, pathogen, treatment, test, anatomy, epidemiology, or clinical fact. "
    "About half of the notes contain an error. Compare each numbered sentence against the rest of the "
    "case context and expected medical knowledge. Answer TRUE if any numbered sentence is medically "
    "inconsistent or likely erroneous. Answer FALSE only if all numbered sentences are medically consistent. "
    "Return only TRUE or FALSE."
)
MEDEC_ERROR_SENTENCE_ID_MAP = (
    "Identify the numbered sentence that contains the medical error. "
    "Return only the sentence ID as an integer, with no explanation."
)
MEDEC_CORRECTED_SENTENCE_MAP = (
    "Generate the corrected version of the erroneous numbered sentence. "
    "Return only the corrected sentence, with no explanation."
)
