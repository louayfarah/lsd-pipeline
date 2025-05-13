# Load query templates
# Infer QA answers
# Infer facts under certain parameters
import re
import pandas as pd
from ollama import Client
import numpy as np
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from util.query_templates import make_fact_prompt, make_mcq_prompt
from validate import validate_answers


def infer_answer(prompt: str, model: str):
    """
    Send `prompt` to Ollama `model`, return (prediction, generation_time_s).
    """
    client = Client()
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
            format={
                "type": "object",
                "properties": {
                    "answer_key": {
                        "type": "string",
                        "description": "The answer key (A, B, C, or D)"
                    }
                },
                "required": ["answer_key"]
            },
            keep_alive=False
        )

        response_dict = vars(resp)

        prompt_eval_time_s = response_dict["prompt_eval_duration"] / 1e9
        eval_time_s = response_dict["eval_duration"] / 1e9

        answer = None

        if hasattr(resp, 'message'):
            response_dict['message'] = vars(resp.message)
            if response_dict.get("message").get("content"):
                content = response_dict["message"]["content"]
                m = re.search(r"\b([ABCD])\b", content, re.IGNORECASE)
                answer = m.group(1).upper() if m else None
        

        return {
            "answer_key": answer, 
            "prompt_eval_time_s": prompt_eval_time_s,
            "eval_time_s": eval_time_s,
        }
    finally:
        pass


def infer_fact(prompt: str, model: str, max_tokens: int, temperature: float):
    """
    Generate a supporting fact under given decoding params.
    Returns (fact_text, gen_time_s).
    """
    client = Client()
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={
                "num_predict": max_tokens,
                "temperature": temperature
            },
            keep_alive=False
        )
        response_dict = vars(resp)

        prompt_eval_time_s = response_dict["prompt_eval_duration"] / 1e9
        eval_time_s = response_dict["eval_duration"] / 1e9

        answer = None

        if hasattr(resp, 'message'):
            response_dict['message'] = vars(resp.message)
            if response_dict.get("message").get("content"):
                content = response_dict["message"]["content"]
                answer = content.strip()

        return {
            "fact_text": answer,
            "prompt_eval_time_s": prompt_eval_time_s,
            "eval_time_s": eval_time_s,
        }
    finally:
        pass



def run_mcq_inference(row, model):
    # parse the choices column
    obj = eval(row["choices"], {"array": np.array})
    texts = obj["text"].tolist()
    labels = obj["label"].tolist()
    choices = dict(zip(labels, texts))

    prompt = make_mcq_prompt(row["question_stem"], choices)

    if model == "llama":
        model = "llama3.2:3b"
    elif model == "deepseek":
        model = "deepseek-r1:7b"

    inferred = infer_answer(prompt, model)
    ans = inferred["answer_key"]
    prompt_eval_time = inferred["prompt_eval_time_s"]
    eval_time = inferred["eval_time_s"]
    return ans, prompt_eval_time, eval_time


def run_fact_inference(row, model, ans, settings):
    # parse the choices column
    obj = eval(row["choices"], {"array": np.array})
    texts = obj["text"].tolist()
    labels = obj["label"].tolist()
    choices = dict(zip(labels, texts))

    prompt = make_fact_prompt(row["question_stem"], choices, ans)

    if model == "llama":
        model = "llama3.2:3b"
    elif model == "deepseek":
        model = "deepseek-r1:7b"

    inferred = infer_fact(prompt, model, settings["max_tokens"], settings["temperature"])
    fact = inferred["fact_text"]
    prompt_eval_time = inferred["prompt_eval_time_s"]
    eval_time = inferred["eval_time_s"]
    return fact, prompt_eval_time, eval_time, len(prompt.split())


def run_inference(sample_csv, output_csv):
    try:
        df = pd.read_csv(sample_csv)
        for model in ("llama", "deepseek"):
            df[f"{model}_pred"]   = None
            df[f"{model}_prompt_eval_time_s"] = None
            df[f"{model}_eval_time_s"] = None
            df[f"{model}_fact_long_low_temp"] = None
            df[f"{model}_fact_long_low_temp_prompt_eval_time_s"] = None
            df[f"{model}_fact_long_low_temp_eval_time_s"] = None
            df[f"{model}_fact_long_high_temp"] = None
            df[f"{model}_fact_long_high_temp_prompt_eval_time_s"] = None
            df[f"{model}_fact_long_high_temp_eval_time_s"] = None
            df[f"fact_prompt_number_tokens"] = None

        for model in ("llama", "deepseek"):
            row_0 = df.iloc[0]
            _, _, _ = run_mcq_inference(row_0, model)
            for idx, row in df.iterrows():
                # run MCQ inference
                ans, prompt_eval_time, eval_time = run_mcq_inference(row, model)
                df.at[idx, f"{model}_pred"]   = ans
                df.at[idx, f"{model}_prompt_eval_time_s"] = prompt_eval_time
                df.at[idx, f"{model}_eval_time_s"] = eval_time

                progress = (idx + 1) / len(df) * 100
                print(f"{model}: Answer keys infered for row {idx}, progress: {progress:.2f}%")
                print(f"{model}: Answer key: {ans}")
        
        df.to_csv(output_csv, index=False)
        print(f"Saved temporary results to {output_csv}")

        validated_df = validate_answers(df)
        for model in ("llama", "deepseek"):
            row_0 = validated_df.iloc[0]
            _, _, _, _ = run_fact_inference(row_0, model, ans, {"max_tokens": 20, "temperature": 0.1})
            for idx, row in validated_df.iterrows():
                # run fact inference
                # long, low temp
                settings = {
                    "max_tokens": 200,
                    "temperature": 0.1
                }
                fact, prompt_eval_time, eval_time, fact_prompt_number_tokens = run_fact_inference(row, model, ans, settings)
                df.at[idx, f"{model}_fact_long_low_temp"] = fact
                df.at[idx, f"{model}_fact_long_low_temp_prompt_eval_time_s"] = prompt_eval_time
                df.at[idx, f"{model}_fact_long_low_temp_eval_time_s"] = eval_time

                # long, high temp
                settings = {
                    "max_tokens": 200,
                    "temperature": 0.9
                }
                fact, prompt_eval_time, eval_time, fact_prompt_number_tokens = run_fact_inference(row, model, ans, settings)
                df.at[idx, f"{model}_fact_long_high_temp"] = fact
                df.at[idx, f"{model}_fact_long_high_temp_prompt_eval_time_s"] = prompt_eval_time
                df.at[idx, f"{model}_fact_long_high_temp_eval_time_s"] = eval_time
                df.at[idx, f"fact_prompt_number_tokens"] = fact_prompt_number_tokens
            
                progress = (idx + 1) / len(df) * 100
                print(f"{model}: Facts with different variations infered for row {idx}, progress: {progress:.2f}%")
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing row: {e}")
    finally:
        df.to_csv(output_csv, index=False)
        print(f"Saved all results to {output_csv}")

run_inference(
    "data/openbook_qa_en_sample100_by_complexity.csv",
    "data/openbook_qa_en_inferred.csv"
)
run_inference(
    "data/openbook_qa_fr_sample100_by_complexity.csv",
    "data/openbook_qa_fr_inferred.csv"
)
