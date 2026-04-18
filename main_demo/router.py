"""
Router Module for MALMAS

Implements the Router Agent described in Section 4.1.1 of the paper:
"At iteration r, a Router Agent selects an active subset A(r) ⊆ A,
and only the selected agents run in parallel to explore complementary
feature interactions, transformations, and compositions."
"""

import os
import json
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime


# Default router prompt path
DEFAULT_ROUTER_PROMPT_PATH = "prompt_files/router.txt"


def load_router_prompt(prompt_path: str = DEFAULT_ROUTER_PROMPT_PATH) -> str:
    """Load the router prompt template from file."""
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


class Router:
    """
    Router Agent that selects active agent subset for each iteration.
    
    Selection strategies:
    1. Data-driven: Select agents based on dataset characteristics
    2. Performance-driven: Select agents based on historical gains
    3. Hybrid: Combine both strategies
    """
    
    # Agent capabilities mapping
    AGENT_CAPABILITIES = {
        "unaryfeature": {
            "description": "Generates features from single columns",
            "required_column_types": ["numerical", "categorical"],
            "excluded_if": ["no_single_column_features"]
        },
        "crosscompositional": {
            "description": "Generates cross features between 2+ columns",
            "required_column_types": ["numerical", "categorical"],
            "min_columns": 2,
            "excluded_if": ["single_column_dataset"]
        },
        "aggregationconstruct": {
            "description": "Generates aggregation-based features",
            "required_column_types": ["categorical", "groupable"],
            "excluded_if": ["no_categorical_for_grouping"]
        },
        "temporalfeature": {
            "description": "Generates time-based features",
            "required_column_types": ["datetime", "temporal"],
            "excluded_if": ["no_datetime_columns"]
        },
        "localtransform": {
            "description": "Generates local transformation features",
            "required_column_types": ["numerical"],
            "excluded_if": ["no_numerical_columns"]
        },
        "localpattern": {
            "description": "Generates features based on distributional patterns",
            "required_column_types": ["numerical", "categorical"],
            "requires_enrich": True,
            "excluded_if": []
        }
    }
    
    def __init__(
        self,
        prompt_path_list: List[str],
        strategy: str = "hybrid",
        min_agents: Optional[int] = None,
        max_agents: Optional[int] = None,
        performance_threshold: float = 0.0,
        warmup_rounds: int = 1,
        use_llm: bool = False,
        router_prompt_path: str = DEFAULT_ROUTER_PROMPT_PATH
    ):
        """
        Initialize Router.

        Args:
            prompt_path_list: List of paths to agent prompt files
            strategy: Selection strategy - "data_driven", "performance_driven", or "hybrid"
            min_agents: Minimum number of agents to select per round (None = no minimum)
            max_agents: Maximum number of agents to select per round (None = no maximum, i.e., can select all)
            performance_threshold: Minimum average gain for an agent to be selected
            warmup_rounds: Number of initial rounds to use all agents
            use_llm: Whether to use LLM for agent selection (requires generate_response function)
            router_prompt_path: Path to the router prompt file
        """
        self.agent_names = [
            os.path.splitext(os.path.basename(p))[0]
            for p in prompt_path_list
        ]
        self.prompt_path_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in prompt_path_list
        }
        self.strategy = strategy
        # None means no constraint - can select from 0 to all agents
        self.min_agents = min_agents if min_agents is not None else 0
        self.max_agents = max_agents if max_agents is not None else len(self.agent_names)
        self.performance_threshold = performance_threshold
        self.warmup_rounds = warmup_rounds
        self.use_llm = use_llm
        self.router_prompt_path = router_prompt_path

        # Load router prompt template
        self.router_prompt_template = load_router_prompt(router_prompt_path)

        # Performance tracking
        self.agent_performance: Dict[str, List[float]] = {
            name: [] for name in self.agent_names
        }
        self.agent_selection_count: Dict[str, int] = {
            name: 0 for name in self.agent_names
        }

        # Dataset characteristics (set during first call)
        self.dataset_characteristics: Optional[Dict[str, Any]] = None
        
    def analyze_dataset(
        self,
        df: pd.DataFrame,
        description: Dict[str, Any],
        enrich_description: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze dataset characteristics to inform agent selection.
        
        Args:
            df: Training dataframe
            description: Column description dictionary
            enrich_description: Enriched description (for localpattern agent)
            
        Returns:
            Dictionary of dataset characteristics
        """
        characteristics = {
            "total_columns": len(df.columns),
            "numerical_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "has_enrich_description": enrich_description is not None,
            "target_column": None
        }
        
        for col_name, col_info in description.items():
            if isinstance(col_info, dict):
                col_type = col_info.get("type", "unknown").lower()
                if col_type in ["numerical", "numeric", "float", "int", "integer"]:
                    characteristics["numerical_columns"].append(col_name)
                elif col_type in ["categorical", "category", "string", "object"]:
                    characteristics["categorical_columns"].append(col_name)
                elif col_type in ["datetime", "date", "time", "temporal"]:
                    characteristics["datetime_columns"].append(col_name)
        
        return characteristics
    
    def _data_driven_selection(self) -> List[str]:
        """
        Select agents based on dataset characteristics.

        Returns:
            List of selected agent names
        """
        if self.dataset_characteristics is None:
            # Fallback: select all agents if no dataset info
            return self.agent_names[:self.max_agents] if self.max_agents < len(self.agent_names) else self.agent_names

        selected = []
        chars = self.dataset_characteristics

        for agent_name in self.agent_names:
            capabilities = self.AGENT_CAPABILITIES.get(agent_name, {})
            should_include = True

            # Check exclusions
            excluded_if = capabilities.get("excluded_if", [])

            if "no_datetime_columns" in excluded_if and not chars["datetime_columns"]:
                should_include = False
            if "single_column_dataset" in excluded_if and chars["total_columns"] <= 2:
                should_include = False
            if "no_numerical_columns" in excluded_if and not chars["numerical_columns"]:
                should_include = False
            if "no_categorical_for_grouping" in excluded_if and len(chars["categorical_columns"]) < 1:
                should_include = False
            if "requires_enrich" in capabilities and not chars["has_enrich_description"]:
                should_include = False

            if should_include:
                selected.append(agent_name)

        # Ensure minimum agents (if min_agents > 0)
        if self.min_agents > 0 and len(selected) < self.min_agents:
            # Add agents from the full list that aren't excluded
            for agent in self.agent_names:
                if agent not in selected and len(selected) < self.min_agents:
                    selected.append(agent)

        # Limit to maximum (if max_agents is set)
        if self.max_agents < len(self.agent_names):
            return selected[:self.max_agents]
        return selected
    
    def _performance_driven_selection(self) -> List[str]:
        """
        Select agents based on historical performance (gains).

        Returns:
            List of selected agent names
        """
        if not any(self.agent_performance.values()):
            # No performance data yet, select all
            return self.agent_names[:self.max_agents] if self.max_agents < len(self.agent_names) else self.agent_names

        # Calculate average performance for each agent
        avg_performance = {}
        for agent_name, gains in self.agent_performance.items():
            if gains:
                avg_performance[agent_name] = sum(gains) / len(gains)
            else:
                avg_performance[agent_name] = float('-inf')  # Agents with no data get lowest priority

        # Sort by performance (descending)
        sorted_agents = sorted(
            avg_performance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top performers above threshold
        selected = []
        for agent_name, avg_gain in sorted_agents:
            if avg_gain >= self.performance_threshold:
                selected.append(agent_name)
            elif self.min_agents > 0 and len(selected) < self.min_agents:
                # Fill to meet minimum if required
                selected.append(agent_name)
            if self.max_agents < len(self.agent_names) and len(selected) >= self.max_agents:
                break

        return selected

    def build_selection_context(
        self,
        round_idx: int,
        description: Optional[Dict[str, Any]] = None,
        task_description: Optional[str] = None
    ) -> str:
        """
        Build the user prompt context for agent selection.

        Args:
            round_idx: Current round index
            description: Column description dictionary
            task_description: Task description text

        Returns:
            Formatted context string for LLM-based selection
        """
        context_parts = []

        # Round information
        context_parts.append(f"Current iteration: Round {round_idx + 1}")

        # Dataset characteristics
        if self.dataset_characteristics:
            chars = self.dataset_characteristics
            context_parts.append(f"\nDataset Characteristics:")
            context_parts.append(f"- Total columns: {chars.get('total_columns', 0)}")
            context_parts.append(f"- Numerical columns: {len(chars.get('numerical_columns', []))}")
            context_parts.append(f"- Categorical columns: {len(chars.get('categorical_columns', []))}")
            context_parts.append(f"- Datetime columns: {len(chars.get('datetime_columns', []))}")
            if chars.get('datetime_columns'):
                context_parts.append(f"  Datetime column names: {', '.join(chars['datetime_columns'])}")

        # Agent performance history
        if any(self.agent_performance.values()):
            context_parts.append(f"\nAgent Performance History (Average Gain):")
            for agent_name in self.agent_names:
                gains = self.agent_performance.get(agent_name, [])
                if gains:
                    avg_gain = sum(gains) / len(gains)
                    context_parts.append(f"- {agent_name}: {avg_gain:.4f} (based on {len(gains)} rounds)")
                else:
                    context_parts.append(f"- {agent_name}: No data yet")

        # Agent selection history
        context_parts.append(f"\nAgent Selection Count (This Experiment):")
        for agent_name, count in self.agent_selection_count.items():
            context_parts.append(f"- {agent_name}: selected {count} time(s)")

        # Agent descriptions
        context_parts.append(f"\nAvailable Agents:")
        for agent_name, caps in self.AGENT_CAPABILITIES.items():
            context_parts.append(f"- {agent_name}: {caps['description']}")

        # Task description if provided
        if task_description:
            context_parts.append(f"\nTask Description: {task_description}")

        return "\n".join(context_parts)

    def _llm_based_selection(
        self,
        round_idx: int,
        description: Optional[Dict[str, Any]] = None,
        task_description: Optional[str] = None
    ) -> List[str]:
        """
        Use LLM to select agents based on the router prompt.

        Args:
            round_idx: Current round index
            description: Column description dictionary
            task_description: Task description text

        Returns:
            List of selected agent names
        """
        try:
            # Import here to avoid circular imports
            from .main_func import generate_response
            import global_config

            user_context = self.build_selection_context(round_idx, description, task_description)

            response = generate_response(
                global_config.LLM.get("llm_model", "deepseek-chat"),
                global_config.LLM.get("api_key", ""),
                global_config.LLM.get("base_url", ""),
                self.router_prompt_template,
                user_context,
                temperature=0.3  # Lower temperature for more deterministic output
            )

            # Parse JSON response
            if response:
                selected = json.loads(response)
                if isinstance(selected, list) and all(isinstance(a, str) for a in selected):
                    # Validate agent names
                    valid_selected = [a for a in selected if a in self.agent_names]
                    if valid_selected:
                        print(f"[Router-LLM] LLM selected: {valid_selected}")
                        return valid_selected

            print("[Router-LLM] Failed to parse LLM response, falling back to rule-based")
        except Exception as e:
            print(f"[Router-LLM] Error: {e}")

        # Fallback to hybrid selection
        return self._hybrid_selection()

    def _hybrid_selection(self) -> List[str]:
        """Internal hybrid selection method."""
        data_selected = set(self._data_driven_selection())
        perf_selected = set(self._performance_driven_selection())

        # Union of both, but prioritize those in both sets
        in_both = list(data_selected & perf_selected)
        only_data = list(data_selected - perf_selected)
        only_perf = list(perf_selected - data_selected)

        selected_names = in_both + only_data + only_perf

        # Apply max constraint if set
        if self.max_agents < len(self.agent_names):
            selected_names = selected_names[:self.max_agents]

        # Apply min constraint if set
        if self.min_agents > 0 and len(selected_names) < self.min_agents:
            remaining = [a for a in self.agent_names if a not in selected_names]
            selected_names.extend(remaining[:self.min_agents - len(selected_names)])

        return selected_names

    def select_agents(
        self,
        round_idx: int,
        df: Optional[pd.DataFrame] = None,
        description: Optional[Dict[str, Any]] = None,
        enrich_description: Optional[Dict[str, Any]] = None,
        task_description: Optional[str] = None
    ) -> List[str]:
        """
        Select active agent subset for the current round.

        Args:
            round_idx: Current round index (0-based)
            df: Training dataframe (for first-round analysis)
            description: Column description dictionary
            enrich_description: Enriched description
            task_description: Task description text (for LLM-based selection)

        Returns:
            List of selected agent prompt paths
        """
        # Warmup: use all agents for initial rounds
        if round_idx < self.warmup_rounds:
            selected_names = self.agent_names
            print(f"[Router] Round {round_idx + 1}: Warmup - using all {len(selected_names)} agents")
        else:
            # Analyze dataset on first non-warmup round if not done
            if self.dataset_characteristics is None and df is not None:
                self.dataset_characteristics = self.analyze_dataset(
                    df, description or {}, enrich_description
                )
                print(f"[Router] Dataset analysis: {len(self.dataset_characteristics.get('numerical_columns', []))} numerical, "
                      f"{len(self.dataset_characteristics.get('categorical_columns', []))} categorical, "
                      f"{len(self.dataset_characteristics.get('datetime_columns', []))} datetime columns")

            # Apply selection strategy
            if self.use_llm and self.router_prompt_template:
                # Use LLM-based selection
                selected_names = self._llm_based_selection(
                    round_idx, description, task_description
                )
            elif self.strategy == "data_driven":
                selected_names = self._data_driven_selection()
            elif self.strategy == "performance_driven":
                selected_names = self._performance_driven_selection()
            else:  # hybrid
                selected_names = self._hybrid_selection()

            print(f"[Router] Round {round_idx + 1}: Selected {len(selected_names)}/{len(self.agent_names)} agents - {selected_names}")

        # Update selection counts
        for name in selected_names:
            self.agent_selection_count[name] += 1
        
        # Return prompt paths
        return [self.prompt_path_map[name] for name in selected_names]
    
    def update_performance(
        self,
        agent_name: str,
        gain: float
    ):
        """
        Update performance history for an agent.
        
        Args:
            agent_name: Name of the agent
            gain: Performance gain (e.g., AUC improvement)
        """
        if agent_name in self.agent_performance:
            self.agent_performance[agent_name].append(gain)
            # Keep only recent performance (last 10 rounds)
            self.agent_performance[agent_name] = self.agent_performance[agent_name][-10:]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get router summary statistics.
        
        Returns:
            Dictionary with selection counts and performance stats
        """
        summary = {
            "selection_counts": self.agent_selection_count.copy(),
            "average_performance": {
                name: sum(gains) / len(gains) if gains else 0
                for name, gains in self.agent_performance.items()
            },
            "strategy": self.strategy,
            "dataset_characteristics": self.dataset_characteristics
        }
        return summary
