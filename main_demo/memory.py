import os
import json
from datetime import datetime
import pandas as pd
import traceback
from .path_helper import add_base_to_sys_path
add_base_to_sys_path(2)
import global_config
from .main_func import *


class AgentMemory:
    def __init__(self, agent_name: str, project_name: str, cache_dir:str , Nround: int = 0):
        self.agent_name = agent_name
        self.project_name = project_name
        
        self.memory_path = os.path.join(cache_dir, f"{Nround}_{agent_name}_memory.json")
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        self.stats = {}
        self.conceptual_summary = ""
        self.memory = {
            "procedural": [],
            "feedback": [],
            "conceptual": [],
            "unused_procedural": [],
            "global_summary":[]
            
        }
        self._load_memory()

    def __del__(self):
        self.save_memory()

    # === 1. 读写操作 ===
    def _load_memory(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self.memory = json.load(f)

    def save_memory(self):
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    # === 2. 操作性记忆 ===
    def record_procedure(self, base_columns, transform: str, feature_name: str, ty: str, description: str, round_idx: int):
        if any(item["feature_name"] == feature_name for item in self.memory["procedural"]):
            return
        self.memory["procedural"].append({
            "base_columns": base_columns,
            "transform": transform,
            "feature_name": feature_name,
            "type": ty,
            "description": description,
            "agent_name": self.agent_name,
            "round_idx": round_idx
        })

    def record_unused_procedure(self, base_columns, transform: str, feature_name: str, ty: str, description: str, round_idx: int):
        if any(item["feature_name"] == feature_name for item in self.memory["unused_procedural"]):
            return
        self.memory["unused_procedural"].append({
            "base_columns": base_columns,
            "transform": transform,
            "feature_name": feature_name,
            "type": ty,
            "description": description,
            "agent_name": self.agent_name,
            "round_idx": round_idx
        })

    # def summarize_procedural(self):
    #     return "\n".join([
    #         f"字段: {item['field']} → 变换: {item['transform']} → 特征: {item['feature_name']}"
    #         for item in self.memory["procedural"]
    #     ])

    # === 3. 反馈性记忆 ===
    def record_feedback(self, feature_name: str, metric: str, value: float, effective: bool, round_idx: int, agent_name: str, base: list, ty: str):
        if any(item["feature_name"] == feature_name for item in self.memory["feedback"]):
            return
        self.memory["feedback"].append({
            "feature_name": feature_name,
            "metric": metric,
            "value": value,
            "effective": effective,
            "round_idx": round_idx,
            "agent_name": agent_name,
            "base_columns": base,
            "type": ty
        })

    def summarize_feedback(self, top_k=5):
        if not self.memory.get("feedback"):
            return ""
        sorted_feedback = sorted(
            [fb for fb in self.memory["feedback"] if fb.get("effective", False)],
            key=lambda x: x["value"],
            reverse=True
        )[:top_k]
        return "\n".join([
            f"{fb['feature_name']} → {fb['metric']}: {fb['value']:.4f}（rank {i+1}）"
            for i, fb in enumerate(sorted_feedback)
        ])

    # === 4. 概念性记忆 ===
    def record_conceptual(self, rule: str):
        if rule not in self.memory["conceptual"]:
            self.memory["conceptual"].append(rule)

    def summarize_conceptual(self):
        return "\n".join(self.memory["conceptual"])

    # === 5. Prompt 插入接口 ===
    def generate_prompt_section(self, use_procedural=False, use_feedback=True,
                                # use_conceptual=True
                               ):
        sections = []
        if use_feedback and self.memory["feedback"]:
            sections.append("【history feedback】\n" + self.summarize_feedback())


        if use_procedural and self.memory["procedural"]:
            sections.append("【Operation attempted】\n" + self.summarize_procedural())
            
        # if use_conceptual and self.memory["conceptual"]:
        #     sections.append("【conceptual memory】\n" + self.summarize_conceptual())
        return "\n\n".join(sections)

    # === 6. 控制策略：何时使用记忆 ===
    def should_use_memory(self, round_idx, warmup_round=1):
        return round_idx >= warmup_round

    def get_positive_negative_columns(self):
        positive = [
            fb["feature_name"] for fb in self.memory.get("feedback", [])
            if fb.get("effective", False)
        ]
        negative = [
            item["feature_name"] for item in self.memory.get("unused_procedural", [])
        ]
        return positive, negative

    def mechanical_summary_for_conceptual(self, min_effective=1):
        effective_transforms = {}
        effective_fields = {}
        effective_types = {}

        for fb in self.memory.get("feedback", []):
            if not fb.get("effective", False):
                continue
            proc = next(
                (p for p in self.memory["procedural"] if p["feature_name"] == fb["feature_name"]),
                None
            )
            if proc:
                tf = proc["transform"]
                effective_transforms[tf] = effective_transforms.get(tf, 0) + 1
                base_fields = "-".join(sorted(proc["base_columns"]))
                effective_fields[base_fields] = effective_fields.get(base_fields, 0) + 1
                ty = proc.get("type", "unknown")
                effective_types[ty] = effective_types.get(ty, 0) + 1

        effective_transforms = {k: v for k, v in effective_transforms.items() if v >= min_effective}
        effective_fields = {k: v for k, v in effective_fields.items() if v >= min_effective}
        effective_types = {k: v for k, v in effective_types.items() if v >= min_effective}

        self.stats = {
            "effective_transforms": effective_transforms,
            "effective_fields": effective_fields,
            "effective_types": effective_types
        }
        return self.stats

    def generate_conceptual_summary_llm(self, min_effective=1):
        num_effective = sum(1 for fb in self.memory.get("feedback", []) if fb.get("effective", False))
        if num_effective < min_effective:
            print(f"⚠️ [{self.agent_name}]Effective feature count {num_effective} is less than min_effective({min_effective}), skipping LLM-generated conceptual summary.")
            self.conceptual_summary = "Few valid features, no available information at the moment."

        else:
            print(f"📝 [{self.agent_name}] Generating local conceptual summary ...")
            stats = self.mechanical_summary_for_conceptual(min_effective=min_effective)
            effective_examples = []
            for fb in self.memory.get("feedback", []):
                if not fb.get("effective", False):
                    continue
                proc = next(
                    (p for p in self.memory["procedural"] if p["feature_name"] == fb["feature_name"]),
                    None
                )
                if proc:
                    example = {
                        "feature_name": fb["feature_name"],
                        "transform": proc["transform"],
                        "base_columns": proc["base_columns"],
                        "type": proc.get("type", "unknown"),
                        "gain": fb["value"],
                        "round_idx": fb.get("round_idx", -1),
                        "agent_name": fb.get("agent_name", self.agent_name)
                    }
                    effective_examples.append(example)

            if not effective_examples:
                return "No effective patterns were found in this agent's recent feature generation."

            examples_text = json.dumps(effective_examples, ensure_ascii=False, indent=2)
            stats_text = json.dumps(stats, ensure_ascii=False, indent=2)

            system_prompt = (
                f"You are {self.agent_name} agent, an expert feature engineering assistant be good at \"{self.agent_name}\" feature generation."
                "You will receive a list of effective features and statistics about their patterns. "
                "Your task is to generate effective, high-quality conceptual rules using concise language that can guide future feature generation. "
                "Rules should directly reflect the statistics and examples. Avoid any irrelevant information."
            )
            user_prompt = (
                f"Here are the effective feature examples:\n\n{examples_text}\n\n"
                f"Here are the statistics about effective features:\n\n{stats_text}\n\n"
                "Based on both the examples and the statistics, summarize 1 to 3 concise and actionable conceptual rules to optimize future feature generation. "
                "Rules should be in clear bullet points."
            )

            self.conceptual_summary = generate_response(
                global_config.LLM["llm_model"],
                global_config.LLM["api_key"],
                global_config.LLM["base_url"],
                system_prompt,
                user_prompt,
                0.6
            )
        self.record_conceptual(rule=self.conceptual_summary)
        return self.conceptual_summary

    @staticmethod
    def generate_global_conceptual_summary(memories, task_description):
        print(f"\n📝  Generating global conceptual summary...")
        sections = []
        for agent_name, memory in memories.items():
            conceptual = memory.conceptual_summary.strip() if memory.conceptual_summary else "No conceptual summary available."
            stats = json.dumps(memory.stats, ensure_ascii=False, indent=2) if memory.stats else "No stats available."
            section = (
                f"Agent: {agent_name}:\n"
                f"Statistics:\n{stats}\n"
                f"Conceptual Summary:\n{conceptual}\n"
                "------------------------"
            )
            sections.append(section)

        combined_prompt = "\n\n".join(sections)

        system_prompt = (
            "You are a senior AutoML optimization assistant. "
            "You will receive conceptual summaries and statistics from multiple feature engineering agents. "
            "Your task is to synthesize these into 2 to 5 concise, effective, high-level conceptual rules "
            "that can guide future global feature derivation tasks across all agents. Avoid any irrelevant information."
        )
        user_prompt = (
            f"The description of this dataset is:\n{task_description}"
            f"Here are the conceptual summaries and statistics from all agents:\n\n{combined_prompt}\n\n"
            "Based on the above, summarize 2 to 5 concise, actionable, high-level conceptual rules for optimizing future feature generation across all agents. "
            "Rules should be in clear bullet points."
        )

        global_summary = generate_response(
            global_config.LLM["llm_model"],
            global_config.LLM["api_key"],
            global_config.LLM["base_url"],
            system_prompt,
            user_prompt,
            0.6
        )
        for agent_name, memory_agent in memories.items():
            memory_agent.memory["global_summary"].append(global_summary)
        
        return global_summary
