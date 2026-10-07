from .base import OpName
from vllm.kalypso.budget import KVMemoryManager


SYSTEM_PROMPT = (
    "You are a helpful assistant for executing semantic operators.\n"
    "You will be given data and an operation description.\n"
    "Apply the operation to the provided data exactly as specified and return only the required result.\n"
)


def _build_operation_prompt(instruction, op=OpName.SEM_FILTER):
    if op == OpName.SEM_FILTER:
        return (
            "You will be presented with a context and a filter condition. Output TRUE if the context satisfies the filter condition, and FALSE otherwise.\n"
            "Remember, your answer must be TRUE or FALSE. Finish your response with a newline character\n"
            "Output TRUE or FALSE only.\n"
            f"Condition:{instruction}\n"
        )
    if op == OpName.SEM_MAP:
        return (
            "You  are presented with a context and a mapping instruction.\n"
            "Apply the instruction to the context and produce the mapped output.\n"
            "The output must strictly follow the instruction and contain no extra commentary.\n"
            f"Map Instruction:{instruction}\n"
        )
    if op == OpName.SEM_AGG:
        return (
            "You are presented with multiple contexts.\n"
            "Aggregate them according to the aggregation instruction.\n"
            "The output must be a single aggregated result.\n"
            "Do not include explanations or commentary.\n"
            f"Instruction:{instruction}\n"
        )
    if op == OpName.SEM_TOPK:
        return (
            "You are presented with two contexts.\n"
            "Compare the two contexts according to the ranking instruction.\n"
            "Choose the better context.\n"
            "Answer with 'A' or 'B' only.\n"
            f"Instruction:{instruction}\n"
        )
    if op == OpName.SEM_JOIN:
        return (
            "You are presented with two contexts.\n"
            "Determine whether the two contexts A, B together satisfy the condition.\n"
            "Remember, your answer must be TRUE or FALSE. Finish your response with a newline character\n"
            "The output must strictly follow the condition and contain no extra commentary.\n"
            f"Condition:{instruction}\n"
        )
    if op in (OpName.SEM_CLASSIFY):
        return (
            "You are presented with a context and a classification instruction.\n"
            "Classify the context into exactly one of the provided groups.\n"
            "The output must be one group name only.\n"
            "Do not include explanations or extra text.\n"
            f"Instruction:{instruction}\n"
        )

    raise ValueError(f"Unsupported semantic operation: {op}")


def get_system_prompt():
    messages = [{"role": "system", "type": "text", "content": SYSTEM_PROMPT}]
    return messages


def get_data_prompt(data, right_data=None):    

    if right_data:
        context = (
            "CONTEXT:\n"
            "  {\n"
            f"    \"text\": {data}\n"
            "  }\n"
        )
        for idx, item in enumerate(right_data):
            context += (
                "\n\n"
                f"CONTEXT:\n"
                "  {\n"
                f"    \"text\": {item}\n"
                "  }\n"
            )
    else:
        context = (
            "CONTEXT:\n"
            "  {\n"
            f"    \"text\": {data}\n"
            "  }\n"
        )

    return [{"role": "user", "type": "text", "content": context}]


def get_task_prompt(instruction, op=OpName.SEM_FILTER):
    operation = _build_operation_prompt(instruction, op=op) 
    return [
        {
            "role": "user",
            "type": "text",
            "content": (
                "TASK:\n"
                f"{operation}\n\n"
                "ANSWER:\n"
            ),
        }
    ]


def get_prompt(instruction, data, op=OpName.SEM_FILTER):
    if 'system' in data[0]['role']:
        return data + get_task_prompt(instruction=instruction, op=op)
    return get_system_prompt() + data + get_task_prompt(instruction=instruction, op=op)


def add_assistant_prompt(prompt, output_text):
    output_template = [{
        "role": "assistant",
        "content": output_text
    }]
    appended_prompt = prompt + output_template
    return appended_prompt, KVMemoryManager.get_instance().apply_chat_template(appended_prompt)
