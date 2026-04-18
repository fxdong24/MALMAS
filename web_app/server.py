"""
MALMAS Web Interface - Simple Version with CSV Upload
"""
import sys
import os
import json
import asyncio
import time
import threading
import tempfile
import shutil
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import pandas as pd

# Add MALMAS-63DB to path
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "MALMAS-63DB")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import global_config


# ===== Global State =====
active_tasks = {}
task_logs = {}


def update_llm_config(llm_model: str, api_key: str, base_url: str, code_temp: float = 0.2):
    """Update global LLM configuration."""
    global_config.LLM["llm_model"] = llm_model
    global_config.LLM["api_key"] = api_key
    global_config.LLM["base_url"] = base_url
    global_config.LLM["code_temp"] = code_temp


# ===== Experiment Runner =====

def run_malmas_from_csv(
    task_id: str,
    csv_path: str,
    target_column: str,
    task: str,
    model_name: str,
    metric: str,
    Nround: int,
    temp: float,
    router_strategy: str,
    router_use_llm: bool
):
    """Run MALMAS experiment from uploaded CSV file."""
    task_logs[task_id] = []

    def log(msg):
        task_logs[task_id].append(msg)
        print(f"[{task_id}] {msg}")

    try:
        log(f"Loading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV")

        # Create a simple dataset class for this CSV
        class TempDataset:
            def read_data(self):
                from sklearn.model_selection import train_test_split
                X = df.drop(columns=[target_column])
                y = df[target_column]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                df_train = pd.concat([X_train, y_train], axis=1)
                df_test = pd.concat([X_test, y_test], axis=1)

                # Generate description
                description = {}
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    if df[col].dtype in ['int64', 'float64']:
                        col_type = "numerical"
                    else:
                        col_type = "categorical"
                    description[col] = {
                        "type": col_type,
                        "mean": float(df[col].mean()) if col_type == "numerical" else None,
                        "std": float(df[col].std()) if col_type == "numerical" else None,
                        "unique": int(df[col].nunique()),
                        "missing": int(df[col].isnull().sum())
                    }

                return (
                    df_train,
                    df_test,
                    target_column,
                    f"Auto-generated task: predict '{target_column}'",
                    description,
                    description
                )

            def get_seed_list(self):
                return [0, 1, 2]

        log(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Update config
        from main_demo.model_factory import set_params
        set_params(
            task=task,
            model_name=model_name,
            task_name="uploaded_csv",
            metric=metric
        )

        log(f"Starting MALMAS experiment (task={task}, Nround={Nround})...")

        from main_demo.pipeline import MALMAS_random_experiments_async

        async def _run():
            return await MALMAS_random_experiments_async(
                task_name="uploaded_csv",
                task=task,
                read_data_class=TempDataset(),
                model_name=model_name,
                metric=metric,
                prompt_path_list=global_config.prompt_path_list,
                Nround=Nround,
                min_effective=2,
                long_memory_feature_num=3,
                temp=temp,
                verbose=False,
                router_strategy=router_strategy,
                router_min_agents=None,
                router_max_agents=None,
                router_warmup_rounds=1,
                router_use_llm=router_use_llm
            )

        result = asyncio.run(_run())

        # Convert result to JSON-serializable format
        result_dict = result.reset_index().to_dict(orient="records")
        task_logs[task_id].append("__RESULT__")
        task_logs[task_id].append(json.dumps(result_dict, default=str))
        log("Experiment completed successfully!")

    except Exception as e:
        log(f"ERROR: {str(e)}")
        import traceback
        log(traceback.format_exc())
    finally:
        active_tasks.pop(task_id, None)


# ===== FastAPI App =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="MALMAS - Upload & Run", lifespan=lifespan)

# Serve static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r") as f:
        return f.read()


@app.post("/api/upload")
async def api_upload_csv(
    file: UploadFile = File(...),
    target_column: str = "",
    task: str = "classification",
    model_name: str = "xgboost",
    metric: str = "auc",
    Nround: int = 4,
    temp: float = 1.0,
    router_strategy: str = "hybrid",
    router_use_llm: bool = False,
    llm_model: str = "deepseek-chat",
    api_key: str = "",
    base_url: str = ""
):
    """Upload CSV and run MALMAS experiment."""
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(temp_dir, file.filename)

    try:
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Auto-detect target column if not specified
        if not target_column:
            df = pd.read_csv(csv_path)
            # Assume last column is target
            target_column = df.columns[-1]

        # Update LLM config
        if api_key:
            update_llm_config(llm_model, api_key, base_url)

        # Start experiment
        task_id = f"malmas_{int(time.time())}"
        active_tasks[task_id] = "running"

        thread = threading.Thread(
            target=run_malmas_from_csv,
            args=(
                task_id, csv_path, target_column, task,
                model_name, metric, Nround, temp,
                router_strategy, router_use_llm
            ),
            daemon=True
        )
        thread.start()

        return {"task_id": task_id, "status": "started", "target_column": target_column}

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/experiment/status")
async def api_get_status():
    return {
        "active_tasks": list(active_tasks.keys()),
        "total_tasks": len(active_tasks)
    }


@app.get("/api/experiment/stream/{task_id}")
async def api_stream_logs(task_id: str):
    """Stream experiment logs via SSE."""
    async def event_generator():
        last_idx = 0
        while True:
            if task_id not in task_logs:
                await asyncio.sleep(0.5)
                continue

            logs = task_logs[task_id]
            new_logs = logs[last_idx:]
            last_idx = len(logs)

            for log_msg in new_logs:
                if log_msg == "__RESULT__":
                    continue
                yield {"event": "log", "data": log_msg}

            # Check if task is complete
            if task_id not in active_tasks:
                yield {"event": "done", "data": "completed"}
                break

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/api/experiment/result/{task_id}")
async def api_get_result(task_id: str):
    """Get experiment result."""
    if task_id not in task_logs:
        raise HTTPException(status_code=404, detail="Task not found")

    logs = task_logs[task_id]
    for i, log in enumerate(logs):
        if log == "__RESULT__" and i + 1 < len(logs):
            return {"result": json.loads(logs[i + 1])}

    return {"result": None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
