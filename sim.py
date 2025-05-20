#!huggingface-cli login
from __future__ import annotations
from transformers import pipeline
# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from transformers import pipeline



# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-2b")
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")


import os
from typing import List, Dict
from huggingface_hub import InferenceClient
import re



class Witness:


    def __init__(self,
                 name: str,
                 system_prompt: str,
                 model: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []      # list of {"role": ..., "content": ...}
        """self.client = InferenceClient(
            model,
            token=os.getenv("HF_API_TOKEN")           # make sure this env‑var is set
        )"""

    # ---- helper for HF prompt formatting ----------
    def _format_prompt(self, user_msg: str):
        """
        Formats a full prompt that includes
        * system prompt
        * prior turns
        * new user message
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_msg})

        # HF text-generation endpoints expect a single string.

        prompt = ""
        for m in messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt

    # ---- produce a reply --------------------------
    def respond(self, user_msg: str, **gen_kwargs):
        prompt = self._format_prompt(user_msg)
        """completion = self.client.text_generation(
            prompt,
            max_new_tokens=256,
            temperature=0.8,
            do_sample=True,
            stream=False,
            **gen_kwargs
        )"""
        generator = pipeline('text-generation', model=model, tokenizer = tokenizer)
        completion = generator(prompt, max_new_tokens=256, temperature=0.8, do_sample = True, **gen_kwargs)[0]['generated_text']
        answer = completion.strip()
        # keep chat memory
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})
        print(self.name, " : ", answer, "\n")


# Two–Lawyer Agents (Defense & Prosecution)




class LawyerAgent:


    def __init__(self,
                 name: str,
                 system_prompt: str,
                 model: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.history: List[Dict[str, str]] = []      # list of {"role": ..., "content": ...}
        """self.client = InferenceClient(
            model,
            token=os.getenv("HF_API_TOKEN")           # make sure this env‑var is set
        )"""

    # ---- helper for HF prompt formatting ----------
    def _format_prompt(self, user_msg: str):
        """
        Formats a full prompt that includes
        * system prompt
        * prior turns
        * new user message
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_msg})

        # HF text-generation endpoints expect a single string.

        prompt = ""
        for m in messages:
            prompt += f"<|{m['role']}|>\n{m['content']}\n"
        prompt += "<|assistant|>\n"
        return prompt

    # ---- produce a reply --------------------------
    def respond(self, user_msg: str, **gen_kwargs):
        prompt = self._format_prompt(user_msg)
        """completion = self.client.text_generation(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            stream=False,
            **gen_kwargs
        )"""
        generator = pipeline('text-generation', model=model, tokenizer = tokenizer)
        completion = generator(prompt, max_new_tokens=256, temperature=0.8,do_sample = True, **gen_kwargs)[0]['generated_text']
        answer = completion.strip()
        # keep chat memory
        # Keep the original answer
        original_answer = answer.strip()

        # Check if the last character is a digit and if it's 0 or 1
        if original_answer[-1].isdigit() and original_answer[-1] in ('0', '1'):
            decision = int(original_answer[-1])
            # Remove the decision from the original answer
            answer = original_answer[:-1].strip()
        else:
            decision = None


        witness_name_match = re.search(r"Witness:\s*([A-Za-z]+)", answer)
        witness_name = witness_name_match.group(1) if witness_name_match else None

        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})


        print(self.name, " : ", answer, "\n")
        return answer, witness_name, decision




# System prompts

DEFENSE_SYSTEM = """
You are **Alex Carter**, lead *defense counsel*.
Goals:
• Protect the constitutional rights of the defendant.
• Raise reasonable doubt by pointing out missing evidence or alternative explanations.
• Be respectful to the Court and to opposing counsel.
Style:
• Crisp, persuasive, grounded in precedent and facts provided.
• When citing precedent: give short case name + year (e.g., *Miranda v. Arizona* (1966)).
Ethics:
• Do not fabricate evidence; admit uncertainty when required.
YOU HAVE TO CALL A WITNESS COMPULSORILY
call upon witnesses as and when necessary, but only if they are relevant to the case.

"""

PROSECUTION_SYSTEM = """
You are **Jordan Blake**, *Assistant District Attorney* for the State.
Goals:
• Present the strongest good‑faith case against the accused.
• Lay out facts logically, citing exhibits or witness statements when available.
• Anticipate and rebut common defense arguments.
Style:
• Formal but plain English; persuasive, with confident tone.
Ethics:
• Duty is to justice, not merely to win. Concede points when ethically required.
YOU HAVE TO CALL A WITNESS COMPULSORILY
call upon witnesses as and when necessary, but only if they are relevant to the case.

"""
JUDGE_SYSTEM = """
You are **Judge Sarah Thompson**, presiding over a criminal trial.
Responsibilities:
• Ensure a fair and orderly courtroom.
• Rule on objections and rulings.
• Deliver the verdict based on the evidence presented.

"""


# the two agents

defense = LawyerAgent("Defense", DEFENSE_SYSTEM)
prosecution = LawyerAgent("Prosecution", PROSECUTION_SYSTEM)
judge = LawyerAgent("Judge", JUDGE_SYSTEM)


# Example dialogue  |  State v. Doe

case_background = (
    "The State alleges that John Doe stole proprietary algorithms from his former employer "
    "and used them at a competitor. The charge is felony theft of trade secrets. "
    "No physical evidence shows direct copying, but server logs indicate large downloads "
    "two days before Doe resigned."
)




judge.respond(f"Very short opening statement to the court as a judge, introduce the details of the case :  {case_background}")
x = 1
while(x!=0):
    # Prosecutor goes first
    answer, next, trig = prosecution.respond(
    f"Put forward your points and statements using evidence and information, take into account everything that has been said till now by both the witnesses and the defense"
    )

    witness1 = Witness("**witness name said by prosecution**",
                       f"You are a witness in a criminal trial. You will be questioned by the prosecution and defense attorneys. You will be asked to answer questions about the case. You will be asked to provide evidence that supports the prosecution's case or contradicts the defense's case.")
    witness1.respond(f"Respond to questions asked by the prosecutor, do not lie or fabricate, answer carefully and truthfully. Prosecutions statement : {answer}")



    #Defense responds
    answerd, nextd, trigd = defense.respond(
    "Respond to the prosecution and the claims and put forward your own counterpoints, take into account everything that has been said till now by both the witnesses and the defense"
    )

    witness2 = Witness("**witness name said by defense**",
                       "You are a witness in a criminal trial. You will be questioned by the prosecution and defense attorneys. You will be asked to answer questions about the case. You will be asked to provide evidence that supports the prosecution's case or contradicts the defense's case.")
    witness1.respond(f"Respond to questions asked by the defense, do not lie or fabricate, answer carefully and truthfully. Defense's statement : {answerd}")
