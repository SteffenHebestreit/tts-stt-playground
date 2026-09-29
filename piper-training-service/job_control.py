"""Which training jobs are running, and what may be done to them.

The job registry in app.py is a display record: a status string that the
endpoints, the training thread and the user's cancel all write. It cannot answer
"is a thread still working on this job?", because ``DELETE /job/{id}`` sets
``cancelled`` immediately while the thread keeps going until its next epoch
boundary. Decisions that must not overlap two runs on one GPU, one checkpoint
directory or one dataset therefore go through this registry, which is keyed on
the run itself: claimed when a job is accepted, released when its runner ends.

Free of torch and the audio stack so the rules can be tested on their own.
"""

from typing import Dict, Optional

from fastapi import HTTPException

# Statuses in which a runner is (or is about to be) working on the job.
ACTIVE_STATUSES = frozenset({"initializing", "processing", "transcribing", "training", "exporting"})

# Statuses no runner will change any more.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class ActiveRuns:
    """The runs currently holding a training slot."""

    def __init__(self, max_concurrent: int = 1):
        """Allow at most *max_concurrent* runs at once (never fewer than one)."""
        self.max_concurrent = max(1, int(max_concurrent))
        self._runs: Dict[str, str] = {}  # job_id -> model_name, oldest first

    def claim(self, job_id: str, model_name: str) -> None:
        """Take a slot for *job_id*, or raise 409 naming the job in the way.

        Called from the event loop with no ``await`` between the check and the
        job being registered, which is what makes it race-free.
        """
        if job_id in self._runs:
            raise HTTPException(
                status_code=409,
                detail=f"Job {job_id} is already running; wait for it to stop before starting it again.",
            )
        for other, other_model in self._runs.items():
            if other_model == model_name:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Job {other} is already working on model '{model_name}'. Two jobs would "
                        f"write the same dataset and voice; wait for it, or stop it with DELETE /job/{other}."
                    ),
                )
        if len(self._runs) >= self.max_concurrent:
            running = next(iter(self._runs))
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Training job {running} ('{self._runs[running]}') is already running, and this "
                    f"service runs {self.max_concurrent} job(s) at a time (TRAINING_MAX_CONCURRENT). "
                    f"Wait for it to finish, or stop it with DELETE /job/{running}."
                ),
            )
        self._runs[job_id] = model_name

    def release(self, job_id: str) -> None:
        """Give the slot back. Safe to call for a job that holds none."""
        self._runs.pop(job_id, None)

    def is_active(self, job_id: str) -> bool:
        """Is a runner still working on this job?"""
        return job_id in self._runs

    def model_of(self, job_id: str) -> Optional[str]:
        """The model name a running job is training, if it is running."""
        return self._runs.get(job_id)

    def job_for_model(self, model_name: str) -> Optional[str]:
        """The running job working on *model_name*, if any."""
        return next((j for j, m in self._runs.items() if m == model_name), None)

    def job_ids(self) -> list:
        """Ids of the running jobs, oldest first."""
        return list(self._runs)

    def __len__(self) -> int:
        """Number of slots in use."""
        return len(self._runs)
