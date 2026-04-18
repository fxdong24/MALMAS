""" Implementation of the llmfe pipeline. """
from __future__ import annotations

# from collections.abc import Sequence
from typing import Any, Tuple, Sequence

from . import code_manipulation
from . import config as config_lib
from . import evaluator
from . import buffer
from . import sampler
from . import profile


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """ Return the name of the function to evolve and of the function to run.

    The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@evaluate.run', '@equation.evolve' respectively.
    The function labeled with '@evaluate.run' is going to evaluate the generated code (like data-diven fitness evaluation).
    The function labeled with '@equation.evolve' is the function to be searched (like 'equation' structure).
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    
    return evolve_functions[0], run_functions[0]



def main(
        specification: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        meta_data: dict,
        max_sample_nums: int | None,
        class_config: config_lib.ClassConfig,
        **kwargs
):
    """ Launch a llmfe experiment.
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem.
        config       : config file.
        meta_data    : the metadata file containing the features.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
    """
    function_to_evolve, function_to_run = _extract_function_names(specification)
    template = code_manipulation.text_to_program(specification)
    database = buffer.ExperienceBuffer(config.experience_buffer, template, function_to_evolve, meta_data)

    # get log_dir and create profiler
    log_dir = kwargs.get('log_dir', None)
    if log_dir is None:
        profiler = None
    else:
        profiler = profile.Profiler(log_dir)

    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            template,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class
        ))

    initial = template.get_function(function_to_evolve).body
    evaluators[0].analyse(initial, island_id=None, version_generated=None, data_input=inputs['data']['inputs'], data_output= inputs['data']['outputs'], profiler=profiler)

    # Set global max sample nums.
    samplers = [sampler.Sampler(database, evaluators, 
                                config.samples_per_prompt, 
                                meta_data=meta_data,
                                max_sample_nums=max_sample_nums,
                                llm_class=class_config.llm_class,
                                config = config)
                                for _ in range(config.num_samplers)]

    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    for s in samplers:
        s.sample(profiler=profiler,**kwargs)