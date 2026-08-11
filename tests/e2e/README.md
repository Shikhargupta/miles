# e2e tests

The training e2e tests are scripts, not pytest cases: the CUDA runner executes each file with
`python3`, and each file's `__main__` runs `prepare()` and then `execute()`.

## Running them on the Kubernetes backend

Neither entry point takes a backend argument, so the backend comes from the environment: given
no config of its own, `execute_train` builds one out of the same `MILES_SCRIPT_<FIELD_NAME_UPPER>`
variables the launch scripts bind. See `docs/advanced/cluster-backend.md` for the backend itself.

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -r workbench -- bash -lc \
  "cd /root/miles && \
   MILES_SCRIPT_CLUSTER_BACKEND=kubernetes \
   MILES_SCRIPT_NAMESPACE=$MILES_NS \
   MILES_SCRIPT_HELM_VALUES=/cluster-storage/infra.yaml \
   python3 tests/e2e/short/test_qwen2.5_0.5B_gsm8k_short.py"
```

- Run it from inside the workbench pod, so the launcher, the command Jobs and the training pods
  agree on what a path means.
- `prepare()` obeys the same choice: a download runs in the workbench pod, and
  `convert_checkpoint` runs as a command Job asking for the GPUs it was told to use.
- `MILES_SCRIPT_RUN_ID` is optional: unset, every launch is stamped with the moment it started, so
  a rerun opens its own release. Set it to relaunch one run in place, and give two files running in
  one namespace two different values.
- Repeatable options are space separated inside one variable, exactly as click reads them:
  `MILES_SCRIPT_HELM_VALUES="/cluster-storage/infra.yaml /cluster-storage/quota.yaml"`.
- Mount the shared storage over the image paths the tests name, so every pod reads the same files.
- Unset `MILES_SCRIPT_CLUSTER_BACKEND` and the same file runs on Ray, unchanged. That is how both
  backends stay covered: one suite, run once per environment, rather than one suite run twice.
