# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import glob
import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import field
from enum import Enum
from pathlib import Path

import hydra
import tomlkit

from nemo_skills.inference.generate import GenerationTask
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_config_path
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


class SupportedAgentFrameworks(str, Enum):
    swe_agent = "swe_agent"
    openhands = "openhands"


# Like nemo_skills.inference.generate.InferenceConfig, except most parameters are not passed by default
# because they may not be supported by all LLM servers or agent frameworks.
# tokens_to_generate is purposefully unlimited by default for SWE-bench.
@nested_dataclass(kw_only=True)
class SweBenchInferenceConfig:
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int | None = None
    top_p: float = 0.95
    min_p: float | None = None
    random_seed: int | None = None
    tokens_to_generate: int | None = None
    repetition_penalty: float | None = None
    top_logprobs: int | None = None


# Converts the parameter names above to the corresponding OpenAI parameter names.
NS_TO_OPENAI_PARAM = {
    # Officially part of the OpenAI Chat Completions API.
    "tokens_to_generate": "max_tokens",
    "top_logprobs": "top_logprobs",
    "random_seed": "seed",
    # Not in the official API, but still supported by some servers, e.g. vllm.
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    # temperature and top_p are passed as separate SWE-agent parameters.
}


# Converts the parameter names above to the corresponding parameters in OpenHands's LLM config.
# https://github.com/All-Hands-AI/OpenHands/blob/main/openhands/core/config/llm_config.py#L12
NS_TO_OPENHANDS_PARAM = {
    # Supported on OpenHands's side. top_k is not OpenAI-compatible and so may break some servers.
    "tokens_to_generate": "max_output_tokens",
    "top_k": "top_k",
    "random_seed": "seed",
    # Not supported by OpenHands. Nemo-Skills will raise an error if they are passed.
    "min_p": None,
    "repetition_penalty": None,
    "top_logprobs": None,
    # temperature and top_p are passed separately.
}


# not inheriting since most parameters are not supported because we don't use our model client here
# TODO: should we fix that?
@nested_dataclass(kw_only=True)
class SweBenchGenerationConfig:
    input_file: str  # Path to the input file with data
    output_file: str  # Where to save the generations

    agent_framework: SupportedAgentFrameworks  # Which agentic framework to use

    # URL of the SWE-agent/OpenHands repo to pass to git clone. If None, will use the official repo
    agent_framework_repo: str | None = None
    agent_framework_commit: str = "HEAD"  # Which commit to use when cloning the SWE-agent/OpenHands repo

    # SWE-agent/OpenHands configuration file path. Can be specified in the same way as ns prompt configs
    # If None, will use the default for the chosen framework
    agent_config: str | None = None
    agent_max_turns: int = 100  # Max iterations for the agent

    # Cache directory for cloned repositories. If None, uses workspace directory + '.swe-bench-cache'
    # This allows sharing cached repos across workers on shared filesystems
    cache_dir: str | None = None

    swebench_tests_timeout: int = 60 * 30  # Timeout for the tests after applying the patch, in seconds

    inference: SweBenchInferenceConfig = field(default_factory=SweBenchInferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    max_samples: int = -1  # If > 0, will stop after generating this many samples. Useful for debugging
    skip_filled: bool = False  # If True, will skip the generations that are already in the output file

    # maximum number of concurrent requests to the server for the async loop
    # if sync loop is used, this is the batch size
    max_concurrent_requests: int = 512
    # chunk the dataset into equal sized parts and index into them
    num_chunks: int | None = None  # if specified, will split the data into chunks and only generate for one chunk
    chunk_id: int | None = None  # if specified, will index the specified chunk only

    # if False, will not add num_generated_tokens and generation_time values.
    # Useful when running judge jobs to keep the original generation statistics
    add_generation_stats: bool = True
    generation_key: str = "generation"
    async_position_key: str = "_async_position"  # key to use for preserving position in async loop in data dict
    dry_run: bool = False

    # if True, will move full generation to _full_generation key and keep cfg.generation_key without thinking tokens
    remove_thinking: bool = False
    thinking_begin: str = "<think>"
    thinking_end: str = "</think>"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_swebench_generation_config", node=SweBenchGenerationConfig)


class SweBenchGenerationTask(GenerationTask):
    def __init__(self, cfg: SweBenchGenerationConfig):
        self.cfg = cfg

        LOG.info(
            "Async loop is maintaining %d generations in parallel. "
            "Use max_concurrent_requests to control the number of concurrent requests.",
            self.cfg.max_concurrent_requests,
        )
        self.semaphore = asyncio.Semaphore(self.cfg.max_concurrent_requests)

        # output_lock will be initialized when async_loop is called
        self.output_lock = None

        # needs to skip completed samples, not used otherwise
        self.cfg.prompt_format = "ns"
        
        # Cache directory for cloned repositories
        # Use configured cache_dir, or default to workspace/.swe-bench-cache for shared access
        if self.cfg.cache_dir:
            self.cache_dir = Path(self.cfg.cache_dir)
        else:
            # Use workspace root (where the script is typically run from) for shared cache
            workspace_root = Path(os.getcwd())
            self.cache_dir = workspace_root / ".swe-bench-cache"
        
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        LOG.info(f"Using cache directory: {self.cache_dir}")
        
        # Setup cloned repositories
        self._setup_cached_repos()

    def log_example_prompt(self, data):
        return

    def setup_prompt(self):
        return

    def setup_llm(self):
        return

    def _setup_cached_repos(self):
        """Clone agent framework and evaluation repositories once to cache directory.
        
        Note: Only clones repos on the host. Dependencies are installed inside the container
        on first run, and the venv is persisted in the mounted directory.
        """
        # Setup SWE-agent repository
        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            agent_repo = self.cfg.agent_framework_repo or "https://github.com/SWE-agent/SWE-agent.git"
            self.swe_agent_dir = self.cache_dir / "SWE-agent"
            
            if not self.swe_agent_dir.exists():
                LOG.info(f"Cloning SWE-agent repository to {self.swe_agent_dir}")
                subprocess.run(
                    ["git", "clone", agent_repo, str(self.swe_agent_dir)],
                    check=True,
                    capture_output=True
                )
            
            # Checkout the specified commit
            LOG.info(f"Checking out SWE-agent commit: {self.cfg.agent_framework_commit}")
            subprocess.run(
                ["git", "-C", str(self.swe_agent_dir), "fetch", "origin"],
                check=True,
                capture_output=True
            )
            subprocess.run(
                ["git", "-C", str(self.swe_agent_dir), "checkout", self.cfg.agent_framework_commit],
                check=True,
                capture_output=True
            )
        
        # Setup OpenHands repository
        elif self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
            agent_repo = self.cfg.agent_framework_repo or "https://github.com/All-Hands-AI/OpenHands.git"
            self.openhands_dir = self.cache_dir / "OpenHands"
            
            if not self.openhands_dir.exists():
                LOG.info(f"Cloning OpenHands repository to {self.openhands_dir}")
                subprocess.run(
                    ["git", "clone", agent_repo, str(self.openhands_dir)],
                    check=True,
                    capture_output=True
                )
            
            # Checkout the specified commit
            LOG.info(f"Checking out OpenHands commit: {self.cfg.agent_framework_commit}")
            subprocess.run(
                ["git", "-C", str(self.openhands_dir), "fetch", "origin"],
                check=True,
                capture_output=True
            )
            subprocess.run(
                ["git", "-C", str(self.openhands_dir), "checkout", self.cfg.agent_framework_commit],
                check=True,
                capture_output=True
            )
        
        # Setup SWE-bench evaluation repository (used by both frameworks)
        self.swe_bench_dir = self.cache_dir / "SWE-bench"
        if not self.swe_bench_dir.exists():
            LOG.info(f"Cloning SWE-bench repository to {self.swe_bench_dir}")
            subprocess.run(
                ["git", "clone", "https://github.com/HeyyyyyyG/SWE-bench.git", str(self.swe_bench_dir)],
                check=True,
                capture_output=True
            )

    def _find_container(self, data_point):
        """Find the container file using multiple strategies.
        
        Tries in order:
        1. Exact match with "__" replaced by "_1776_"
        2. Exact match with "__" replaced by "_s_"
        3. Fuzzy search in container directory for files with either replacement
        
        Returns:
            str: Path to the container file (may not exist if all strategies fail)
        """
        instance_id = data_point["instance_id"]
        container_formatter = data_point["container_formatter"]
        
        # Strategy 1: Try _1776_ replacement (original case and lowercase)
        container_name = container_formatter.format(
            instance_id=instance_id.replace("__", "_1776_")
        )
        if os.path.exists(container_name):
            return container_name
        
        # Try lowercase version
        container_name_lower = container_formatter.format(
            instance_id=instance_id.replace("__", "_1776_").lower()
        )
        if os.path.exists(container_name_lower):
            LOG.info(f"Using _1776_ replacement (lowercase): {container_name_lower}")
            return container_name_lower
        
        # Strategy 2: Try _s_ replacement (original case and lowercase)
        container_name_s = container_formatter.format(
            instance_id=instance_id.replace("__", "_s_")
        )
        if os.path.exists(container_name_s):
            LOG.info(f"Using _s_ replacement: {container_name_s}")
            return container_name_s
        
        # Try lowercase version
        container_name_s_lower = container_formatter.format(
            instance_id=instance_id.replace("__", "_s_").lower()
        )
        if os.path.exists(container_name_s_lower):
            LOG.info(f"Using _s_ replacement (lowercase): {container_name_s_lower}")
            return container_name_s_lower
        
        # Strategy 3: Fuzzy search in container directory
        container_dir = os.path.dirname(container_name)
        if os.path.exists(container_dir):
            # Build search patterns for both replacements
            replaced_id_1776 = instance_id.replace("__", "_1776_")
            replaced_id_s = instance_id.replace("__", "_s_")
            
            # Search for .sif files with either replacement pattern (case-insensitive)
            # Include both original case and lowercase versions
            patterns = [
                os.path.join(container_dir, f"*{replaced_id_1776}*.sif"),
                os.path.join(container_dir, f"*{replaced_id_s}*.sif"),
                os.path.join(container_dir, f"*{replaced_id_1776.lower()}*.sif"),
                os.path.join(container_dir, f"*{replaced_id_s.lower()}*.sif")
            ]
            
            matching_files = []
            for pattern in patterns:
                matching_files.extend(glob.glob(pattern))
            
            if matching_files:
                # Use the first matching file found
                container_path = matching_files[0]
                LOG.info(f"Using fuzzy match: {container_path}")
                return container_path
            else:
                LOG.warning(
                    f"No container found with instance_id replacements "
                    f"'{replaced_id_1776}' or '{replaced_id_s}' in {container_dir}"
                )
        else:
            LOG.warning(f"Container directory {container_dir} does not exist")
        
        # Return the original name as fallback (even though it doesn't exist)
        LOG.warning(f"Using non-existent container path: {container_name}")
        return container_name

    async def _execute_container_command(
        self, data_point, command, expected_file_pattern, mode, max_retries=3, timeout=100000, extra_mounts=None
    ):
        """Execute a command in an Apptainer container with retry logic.
        
        Args:
            extra_mounts: List of tuples (src, dst) for additional bind mounts
        """
        # Find the container using multiple strategies
        container_name = self._find_container(data_point)

        # Create logs directory if it doesn't exist
        logs_dir = self.output_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)
        log_file_path = logs_dir / f"{data_point['instance_id']}_{mode}.log"
        LOG.info("Starting execution of an apptainer command. Logs are available at %s", log_file_path)

        # Fix localhost URLs not working sometimes
        command = f"echo '127.0.0.1 localhost' >/etc/hosts && {command}"

        # Build environment variable flags for Apptainer
        env_flags = ""
        if os.getenv("HF_TOKEN"):
            env_flags += f" --env HF_TOKEN={shlex.quote(os.getenv('HF_TOKEN'))}"
            LOG.info("Passing HF_TOKEN to Apptainer container")
        if os.getenv("HF_HOME"):
            env_flags += f" --env HF_HOME={shlex.quote(os.getenv('HF_HOME'))}"
            LOG.info(f"Passing HF_HOME={os.getenv('HF_HOME')} to Apptainer container")
        if os.getenv("HF_DATASETS_OFFLINE"):
            env_flags += f" --env HF_DATASETS_OFFLINE={shlex.quote(os.getenv('HF_DATASETS_OFFLINE'))}"
            LOG.info("Passing HF_DATASETS_OFFLINE to Apptainer container")
        if os.getenv("TRANSFORMERS_OFFLINE"):
            env_flags += f" --env TRANSFORMERS_OFFLINE={shlex.quote(os.getenv('TRANSFORMERS_OFFLINE'))}"
            LOG.info("Passing TRANSFORMERS_OFFLINE to Apptainer container")

        # Build additional mount flags
        mount_flags = (
            f"--mount type=bind,src=/nemo_run/code,dst=/nemo_run/code "
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount "
        )
        
        if extra_mounts:
            for src, dst in extra_mounts:
                mount_flags += f"--mount type=bind,src={src},dst={dst} "
                LOG.info(f"Mounting {src} to {dst}")

        # Launch Apptainer container and execute the command
        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --no-mount home,tmp,bind-paths "
            f"{mount_flags}"
            f"{env_flags} {container_name} bash -c {shlex.quote(command)}"
        )

        # Retry apptainer command up to max_retries times
        for attempt in range(max_retries):
            try:
                # Stream output to log file as it appears
                with open(log_file_path, "w") as log_file:
                    try:
                        # Create async subprocess
                        process = await asyncio.create_subprocess_shell(
                            apptainer_cmd, stdout=log_file, stderr=log_file
                        )
                        # Wait for completion with timeout
                        await asyncio.wait_for(process.communicate(), timeout=timeout)

                        if process.returncode != 0:
                            raise ValueError(f"Command failed with return code {process.returncode}")

                    except asyncio.TimeoutError:
                        # Kill the process if it's still running
                        if process.returncode is None:
                            process.kill()
                            await process.wait()
                        attempt = max_retries  # Force exit the loop on timeout
                        raise ValueError("Command timed out")

                # Look for the expected file
                pred_files = glob.glob(expected_file_pattern, recursive=True)

                if len(pred_files) == 1:
                    # Success, break out of retry loop
                    return pred_files[0]
                else:
                    raise ValueError(
                        f"Expected exactly one file matching {expected_file_pattern} for {data_point['instance_id']}, "
                        f"found {len(pred_files)}."
                    )
            except Exception:
                if attempt < max_retries - 1:
                    LOG.warning(
                        "Attempt %d failed for instance %s. Retrying...",
                        attempt + 1,
                        data_point["instance_id"],
                    )
                    continue
                else:
                    LOG.error("All %d attempts failed for instance %s", max_retries, data_point["instance_id"])
                    LOG.error("Apptainer command failed. Check logs at: %s", log_file_path)
                    raise ValueError(
                        f"Job failed for {data_point['instance_id']}. Check logs at: {log_file_path}. "
                        f"Expected exactly one file matching {expected_file_pattern}, "
                        f"found {len(pred_files) if 'pred_files' in locals() else 'unknown'}."
                    )

    async def _run_swe_agent(self, data_point, api_base):
        """
        Runs SWE-agent on one instance and evaluates the result in a single container execution.
        Returns a dict with the results.
        """
        if self.cfg.agent_config is None:
            self.cfg.agent_config = "eval/swe-bench/swe-agent/default"
        if self.cfg.agent_framework_repo is None:
            self.cfg.agent_framework_repo = "https://github.com/SWE-agent/SWE-agent.git"

        completion_kwargs = {
            openai_param: getattr(self.cfg.inference, ns_param)
            for ns_param, openai_param in NS_TO_OPENAI_PARAM.items()
            if getattr(self.cfg.inference, ns_param) is not None
        }
        if "top_logprobs" in completion_kwargs:
            completion_kwargs["logprobs"] = True

        swe_agent_cmd = (
            # Install uv if not already available
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "source /root/.local/bin/env && "
            # SWE-agent is already cloned and mounted
            "cd /root/SWE-agent && "
            # Install dependencies only if venv doesn't exist (first run only)
            "if [ ! -d venv ]; then "
            "    echo 'Installing SWE-agent dependencies (first run only)...' && "
            "    uv venv --python 3.12 venv && "
            "    uv pip install -p /root/SWE-agent/venv/bin/python -e . ; "
            "fi && "
            # Run the agent
            f"/root/SWE-agent/venv/bin/python -m sweagent run "
            f"    --config {get_config_path(self.cfg.agent_config)} "
            f"    --agent.model.name hosted_vllm/{self.cfg.server.model} "
            f"    --agent.model.api_base {api_base} "
            f"    --agent.model.temperature {self.cfg.inference.temperature} "
            f"    --agent.model.top_p {self.cfg.inference.top_p} "
            f"    --agent.model.completion_kwargs {shlex.quote(json.dumps(completion_kwargs))} "
            f"    --agent.model.per_instance_call_limit {self.cfg.agent_max_turns} "
            f"    --env.deployment.type local "
            f"    --env.repo.type preexisting "
            f"    --env.repo.repo_name testbed "
            f"    --env.repo.base_commit {data_point['base_commit']} "
            f"    --problem_statement.text {shlex.quote(data_point['problem_statement'])} "
            f"    --problem_statement.id {data_point['instance_id']} && "
            # Convert .pred to .jsonl for evaluation
            f"cd /root/SWE-agent && "
            f"find trajectories -name '{data_point['instance_id']}.pred' -exec sh -c 'cp \"$1\" \"${{1%.pred}}.jsonl\"' _ {{}} \\; && "
            # Copy trajectories first (needed even if no patch)
            f"cp -r trajectories /trajectories_mount/ && "
            # Check if patch exists before running evaluation
            f"PRED_FILE=$(find /root/SWE-agent/trajectories -name '{data_point['instance_id']}.jsonl' | head -1) && "
            f"if [ -f \"$PRED_FILE\" ] && grep -q '\"model_patch\":\\s*\"' \"$PRED_FILE\" 2>/dev/null; then "
            # Patch exists, run evaluation
            "    cd /root/SWE-bench && "
            # Install SWE-bench dependencies only if venv doesn't exist (first run only)
            "    if [ ! -d venv ]; then "
            "        echo 'Installing SWE-bench dependencies (first run only)...' && "
            "        uv venv --python 3.12 venv && "
            "        uv pip install -p /root/SWE-bench/venv/bin/python -e . ; "
            "    fi && "
            # Run evaluation with clean environment
            f"    env -u VIRTUAL_ENV /root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
            f"        --predictions_path \"$PRED_FILE\" "
            f"        --instance_ids {data_point['instance_id']} "
            f"        --run_id eval-outputs "
            f"        --timeout {self.cfg.swebench_tests_timeout} "
            f"        --dataset_name {data_point['dataset_name']} "
            f"        --split {data_point['split']} && "
            "    cp -r logs/run_evaluation/eval-outputs /trajectories_mount/ ; "
            "else "
            # No patch, create a dummy report file
            "    echo 'No patch found, skipping evaluation' && "
            f"    mkdir -p /trajectories_mount/eval-outputs/{data_point['instance_id']} && "
            f"    echo '{{' > /trajectories_mount/eval-outputs/{data_point['instance_id']}/report.json && "
            f"    echo '  \"{data_point['instance_id']}\": {{' >> /trajectories_mount/eval-outputs/{data_point['instance_id']}/report.json && "
            f"    echo '    \"resolved\": false,' >> /trajectories_mount/eval-outputs/{data_point['instance_id']}/report.json && "
            f"    echo '    \"patch_exists\": false,' >> /trajectories_mount/eval-outputs/{data_point['instance_id']}/report.json && "
            f"    echo '    \"patch_successfully_applied\": false' >> /trajectories_mount/eval-outputs/{data_point['instance_id']}/report.json && "
            f"    echo '  }}' >> /trajectories_mount/eval-outputs/{data_point['instance_id']}/report.json && "
            f"    echo '}}' >> /trajectories_mount/eval-outputs/{data_point['instance_id']}/report.json ; "
            "fi"
        )

        # Mount both cached directories
        extra_mounts = [
            (str(self.swe_agent_dir), "/root/SWE-agent"),
            (str(self.swe_bench_dir), "/root/SWE-bench")
        ]

        # Execute combined command - look for the evaluation report as the expected file
        search_path = os.path.join(
            self.output_dir, "eval-outputs", "**", f"{data_point['instance_id']}/report.json"
        )
        
        try:
            report_file = await self._execute_container_command(
                data_point,
                swe_agent_cmd,
                search_path,
                mode="agent_and_eval",
                extra_mounts=extra_mounts,
                timeout=self.cfg.swebench_tests_timeout + 300,  # Extra time for agent + eval
            )
            
            # Read the report
            with open(report_file, "r") as f:
                report_json = json.loads(f.read().strip())
                
        except ValueError as e:
            LOG.error("Failed to execute SWE-agent or evaluation for %s: %s", data_point["instance_id"], e)
            report_json = {
                data_point["instance_id"]: {
                    "resolved": False,
                    "patch_exists": False,
                    "patch_successfully_applied": False,
                }
            }

        # Read the trajectory file
        pred_files = glob.glob(
            os.path.join(self.output_dir / "trajectories", "**", f"{data_point['instance_id']}.jsonl"),
            recursive=True
        )
        
        if pred_files:
            with open(pred_files[0], "r") as f:
                trajectory_dict = json.loads(f.read().strip())
        else:
            LOG.warning("No trajectory file found for %s", data_point["instance_id"])
            trajectory_dict = {
                "model_name_or_path": self.cfg.server.get("model", "unknown"),
                "instance_id": data_point["instance_id"],
                "model_patch": None,
            }

        return {
            "swe-bench-metrics": report_json[data_point["instance_id"]],
            "swe-bench-outputs": trajectory_dict,
            "generation": "",  # required TODO: we should fix this
        }

    async def _run_openhands(self, data_point, api_base):
        """
        Runs OpenHands on one instance.
        Returns the absolute (not mounted) path to a .jsonl file in the SWE-bench evaluation format.
        """
        if self.cfg.agent_config is None:
            self.cfg.agent_config = "eval/swe-bench/openhands/default"
        if self.cfg.agent_framework_repo is None:
            self.cfg.agent_framework_repo = "https://github.com/All-Hands-AI/OpenHands.git"

        # Add parameters to config.toml

        with open(get_config_path(self.cfg.agent_config, config_extension="toml"), "r") as f:
            config = tomlkit.parse(f.read())

        config["llm"]["model"] |= {
            "model": self.cfg.server.model,
            "base_url": api_base,
            "temperature": self.cfg.inference.temperature,
            "top_p": self.cfg.inference.top_p,
        }

        for ns_param, oh_param in NS_TO_OPENHANDS_PARAM.items():
            if getattr(self.cfg.inference, ns_param) is not None:
                if oh_param is not None:
                    config["llm"]["model"][oh_param] = getattr(self.cfg.inference, ns_param)
                else:
                    supported_params = [key for key, value in NS_TO_OPENHANDS_PARAM.items() if value is not None]
                    raise ValueError(
                        f"Inference parameter {ns_param} is not supported by OpenHands. "
                        f"Supported inference parameters: temperature, top_p, {', '.join(supported_params)}."
                    )

        config_str = tomlkit.dumps(config)

        openhands_cmd = (
            # make sure /workspace isn't mounted as a safety precaution
            # (mounting it in the nemo-skills cluster config is ok, just not inside of apptainer specifically)
            "if [ -d /workspace ]; then "
            "    echo 'Exiting because /workspace is mounted.' && "
            "    echo 'Please make sure /workspace is not mounted inside of Apptainer before running OpenHands.' && "
            "    echo 'This is because OpenHands DELETES EVERYTHING in the /workspace folder if it exists.' && "
            "    exit 1; "
            "fi && "
            # OpenHands is already cloned and mounted, install dependencies
            "cd /root && "
            'curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && '
            "bash Miniforge3-$(uname)-$(uname -m).sh -b && "
            'eval "$(/root/miniforge3/bin/conda shell.bash hook)" && '
            "mamba install -y --override-channels conda-forge::python=3.12 conda-forge::nodejs conda-forge::poetry conda-forge::tmux && "
            # OpenHands LocalRuntime uses tmux to manage a bash session. In Apptainer, the real UID (id -ru)
            # can differ from the effective UID (root) due to `su root -`. tmux chooses its default socket
            # path based on the REAL UID, e.g., /tmp/tmux-<real-uid>/default. Below we:
            #  - derive the real UID (fallback to id -u)
            #  - force tmux to use that socket path via TMUX/TMUX_TMPDIR
            #  - ensure the directory exists, has proper ownership and permissions
            #  - start the tmux server idempotently on that exact socket
            # This avoids 'error connecting to /tmp/tmux-<uid>/default' during LocalRuntime startup.
            # Ensure tmux socket directory exists with proper permissions, using real UID when available
            "uid=$(id -ru 2>/dev/null || id -u) && "
            "export TMUX_TMPDIR=/tmp && "
            "export TMUX=/tmp/tmux-$uid/default && "
            "mkdir -p /tmp/tmux-$uid && "
            "chown $uid:$uid /tmp/tmux-$uid || true && "
            "chmod 700 /tmp/tmux-$uid && "
            # Start tmux server on the exact socket path (idempotent)
            "tmux -S /tmp/tmux-$uid/default start-server || true && "
            "cd /root/OpenHands && "
            "export INSTALL_DOCKER=0 && "
            "make build && "
            "poetry run python -m pip install datasets && "
            # set up config files
            f"echo {shlex.quote(config_str)} >config.toml && "
            f"echo \"selected_ids = ['{data_point['instance_id']}']\" >evaluation/benchmarks/swe_bench/config.toml && "
            # set local runtime & force verbose logs
            "export RUNTIME=local && "
            "export LOG_ALL_EVENTS=true && "
            "export LOG_LEVEL=DEBUG && "
            # run the agent
            f"./evaluation/benchmarks/swe_bench/scripts/run_infer.sh "
            f"    llm.model "  # name of llm config section in config.toml
            f"    {self.cfg.agent_framework_commit} "  # openhands commit
            f"    CodeActAgent "  # agent
            f"    1 "  # number of instances
            f"    {self.cfg.agent_max_turns} "  # max agent iterations
            f"    1 "  # number of workers
            f"    {data_point['dataset_name']} "  # dataset name
            f"    {data_point['split']} && "  # dataset split
            # move outputs to the mounted directory
            f"mkdir -p /trajectories_mount/trajectories && "
            f"cp -r evaluation/evaluation_outputs/outputs/*/*/* /trajectories_mount/trajectories/{data_point['instance_id']}"
        )

        # Mount the cached OpenHands directory
        extra_mounts = [(str(self.openhands_dir), "/root/OpenHands")]

        # Execute OpenHands command
        search_path = os.path.join(self.output_dir / "trajectories", "**", data_point["instance_id"], "output.jsonl")
        out_file = await self._execute_container_command(
            data_point, openhands_cmd, search_path, mode="agent", extra_mounts=extra_mounts
        )

        with open(out_file, "r") as f:
            out_dict = json.loads(f.read().strip())

        patch = out_dict["test_result"]["git_patch"]
        if not patch:
            patch = None

        # Create file in the SWE-bench evaluation format
        pred_file = out_file.replace("output.jsonl", "output_for_eval.jsonl")
        with open(pred_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": out_dict["metadata"]["llm_config"]["model"],
                        "instance_id": out_dict["instance_id"],
                        "model_patch": patch,
                    }
                )
            )
        return pred_file

    async def process_single_datapoint(self, data_point, data):
        """Will do all necessary generations to get a single answer for the data point."""
        self.output_dir = Path(self.cfg.output_file).parent

        # TODO: what's the right way to support api models, so that our standard parameters for that can be used?
        # TODO: use self.cfg.server.base_url, etc. Can we pass in API key?

        if "base_url" in self.cfg.server:
            api_base = self.cfg.server.base_url
        else:
            api_base = f"http://{self.cfg.server.host}:{self.cfg.server.port}/v1"

        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            # SWE-agent now returns the complete result dict (agent + eval combined)
            output_dict = await self._run_swe_agent(data_point, api_base)
        elif self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
            # OpenHands still uses separate evaluation
            pred_file = await self._run_openhands(data_point, api_base)
            
            pred_mounted_path = pred_file.replace(str(self.output_dir), "/trajectories_mount")
            with open(pred_file, "r") as f:
                trajectory_dict = json.loads(f.read())

            # Check if the trajectory has an empty patch before running evaluation
            has_patch = trajectory_dict["model_patch"] is not None

            if not has_patch:
                report_json = {
                    data_point["instance_id"]: {
                        "resolved": False,
                        "patch_exists": False,
                        "patch_successfully_applied": False,
                    }
                }
            else:
                # Run full evaluation with streaming output
                swe_bench_cmd = (
                    # Install uv if not already available
                    "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                    "source /root/.local/bin/env && "
                    # SWE-bench is already cloned and mounted
                    "cd /root/SWE-bench && "
                    # Install dependencies only if venv doesn't exist (first run only)
                    "if [ ! -d venv ]; then "
                    "    echo 'Installing SWE-bench dependencies (first run only)...' && "
                    "    uv venv --python 3.12 venv && "
                    "    uv pip install -p /root/SWE-bench/venv/bin/python -e . ; "
                    "fi && "
                    # Run with clean environment to avoid venv contamination
                    f"env -u VIRTUAL_ENV /root/SWE-bench/venv/bin/python -m swebench.harness.run_local_evaluation "
                    f"    --predictions_path {pred_mounted_path} "
                    f"    --instance_ids {data_point['instance_id']} "
                    f"    --run_id eval-outputs "
                    f"    --timeout {self.cfg.swebench_tests_timeout} "
                    f"    --dataset_name {data_point['dataset_name']} "
                    f"    --split {data_point['split']} && "
                    f"cp -r logs/run_evaluation/eval-outputs /trajectories_mount/"
                )

                # Mount the cached SWE-bench directory
                extra_mounts = [(str(self.swe_bench_dir), "/root/SWE-bench")]

                # Execute SWE-bench evaluation command
                search_path = os.path.join(
                    self.output_dir, "eval-outputs", "**", f"{data_point['instance_id']}/report.json"
                )
                # TODO: should we fail on errors here? Seems that json isn't always generated
                try:
                    report_file = await self._execute_container_command(
                        data_point,
                        swe_bench_cmd,
                        search_path,
                        mode="eval",
                        timeout=self.cfg.swebench_tests_timeout + 120,
                        extra_mounts=extra_mounts,
                    )
                except ValueError:
                    LOG.error("Failed to execute SWE-bench evaluation command for %s", data_point["instance_id"])
                    report_json = {
                        data_point["instance_id"]: {
                            "resolved": False,
                            "patch_exists": True,
                            "patch_successfully_applied": False,
                        }
                    }
                    report_file = None

                if report_file is not None:
                    with open(report_file, "r") as f:
                        report_json = json.loads(f.read().strip())

            output_dict = {
                "swe-bench-metrics": report_json[data_point["instance_id"]],
                "swe-bench-outputs": trajectory_dict,
                "generation": "",  # required TODO: we should fix this
            }
        else:
            raise ValueError(
                f"Unsupported agent framework: {self.cfg.agent_framework}. "
                f"Supported frameworks: {', '.join(SupportedAgentFrameworks)}."
            )

        return output_dict


GENERATION_TASK_CLASS = SweBenchGenerationTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_swebench_generation_config")
def swebench_generation(cfg: SweBenchGenerationConfig):
    cfg = SweBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = SweBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    SweBenchGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        swebench_generation()
