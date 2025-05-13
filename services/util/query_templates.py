def make_mcq_prompt(stem: str, choices_dict: dict):
    lines = [f"Question: {stem}", "Choices:"]
    for label, txt in choices_dict.items():
        lines.append(f"  {label}) {txt}")
    lines.append("\nPlease answer with exactly one of: A, B, C, or D. Don't write anything else in the answer. Answer absolutely only with one of the letters A, B, C, or D.")
    lines.append("Answer:")
    return "\n".join(lines)


def make_fact_prompt(stem: str, choices_dict: dict, answer_key: str):
    lines = [f"Question: {stem}", "Choices:"]
    for label, txt in choices_dict.items():
        lines.append(f"  {label}) {txt}")
    lines.append(f"\nAnswer: {answer_key}")
    lines.append("Think step by step for as many steps needed. Build a chain of thoughts in order to explain why this answer is correct.")

    return "\n".join(lines)

