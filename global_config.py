
# global_config.py
data_pre = {
    "test_size":0.4
    ,"random_state":42
    }

prompt_path_list = [
    "prompt_files/unaryfeature.txt",
    "prompt_files/crosscompositional.txt",
    "prompt_files/aggregationconstruct.txt",
    "prompt_files/temporalfeature.txt",
    "prompt_files/localtransform.txt",
    "prompt_files/localpattern.txt"
]

LLM = {
    "code_temp":0.2,
    
    
    "llm_model":"deepseek-chat",
    "api_key":"",
    "base_url":"",
    
    # "llm_model":"qwen-max",
    # "api_key":"",
    # "base_url":"",
    
    
    # "llm_model":"gpt-4o-mini",
    # "llm_model":"",
    # "api_key":"",
    # "base_url":"",
    }

other_model=""
compute_tokens=False
total_tokens=0


KFold_random_state=42

task="classification"
task_name=""
metric="auc"# or acc 

model_name="xgboost"
xgboost = {
    "n_estimators":500,
    "learning_rate":0.02
}
