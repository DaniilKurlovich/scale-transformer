#!/bin/sh
# Starts sshd when the container is handed a PUBLIC_KEY (RunPod sets this from
# the key in your account settings), then hands off to CMD. Without that
# variable nothing changes -- `docker run -it ... scale-transformer` still drops
# you straight into bash with no daemon running.
set -eu

if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    # Truncating, not appending: a container restart would otherwise stack up a
    # duplicate key (and a restart loop, hundreds of them). This file is ours.
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys

    # Host keys are generated per container, never baked into the image -- a
    # shared host key would make every pod off this image impersonate the others.
    ssh-keygen -A

    # An ssh session does not inherit the ENVs set in this Dockerfile, so PATH
    # would miss /opt/venv/bin and `python` would be the system one. /etc/environment
    # is read by PAM for non-interactive sessions too (what Zed and `ssh host cmd`
    # use); profile.d covers interactive login shells.
    touch /etc/environment
    : > /etc/profile.d/10-container-env.sh
    for v in PATH LANG HF_HOME TRITON_CACHE_DIR UV_PROJECT_ENVIRONMENT UV_PYTHON \
             UV_LINK_MODE UV_COMPILE_BYTECODE PYTHONUNBUFFERED PYTHONDONTWRITEBYTECODE \
             CUDA_HOME LD_LIBRARY_PATH NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES; do
        eval "val=\${$v:-}"
        [ -n "$val" ] || continue
        sed -i "/^${v}=/d" /etc/environment
        printf '%s=%s\n' "$v" "$val" >> /etc/environment
        printf 'export %s="%s"\n' "$v" "$val" >> /etc/profile.d/10-container-env.sh
    done

    mkdir -p /var/run/sshd

    # AUTO_TRAIN=1: kick off the training job now, in the background, so a pod
    # trains from the moment it is deployed. sshd stays the foreground process
    # (below), which keeps the pod up after the job ends; the job's output is in
    # /workspace/outputs/logs. TRAIN_CONFIG and TRAIN_ARGS are passed through.
    if [ "${AUTO_TRAIN:-0}" = 1 ] && { [ "$#" -eq 0 ] || [ "$*" = "/bin/bash" ]; }; then
        mkdir -p /workspace/outputs/logs
        echo "entrypoint: AUTO_TRAIN=1 -- starting train.sh ${TRAIN_CONFIG:-} ${TRAIN_ARGS:-}" >&2
        # shellcheck disable=SC2086  # TRAIN_ARGS is a list of CLI overrides
        nohup train.sh ${TRAIN_CONFIG:-} ${TRAIN_ARGS:-} \
            >>/workspace/outputs/logs/auto_train.out 2>&1 &
    fi

    # sshd has to be the foreground process. The default CMD is an interactive
    # bash, which on a pod has no TTY and no stdin: it reads EOF, exits, takes
    # PID 1 with it and the pod restart-loops. An explicit command (docker run
    # ... python train.py) still wins -- sshd just goes to the background.
    if [ "$#" -eq 0 ] || [ "$*" = "/bin/bash" ]; then
        exec /usr/sbin/sshd -D -e
    fi
    /usr/sbin/sshd
fi

# Same trap without a key: an interactive bash with nothing on stdin exits at
# once and restart-loops the pod. Locally there is a TTY and this is skipped.
if { [ "$#" -eq 0 ] || [ "$*" = "/bin/bash" ]; } && [ ! -t 0 ]; then
    if [ "${AUTO_TRAIN:-0}" = 1 ]; then
        # No ssh, but still train: the job is the foreground process and its
        # exit code is the container's.
        # shellcheck disable=SC2086
        exec train.sh ${TRAIN_CONFIG:-} ${TRAIN_ARGS:-}
    fi
    echo "entrypoint: PUBLIC_KEY unset -- no sshd. Add an SSH key in your RunPod" >&2
    echo "entrypoint: account settings and redeploy. Idling to keep the pod up." >&2
    exec sleep infinity
fi

exec "$@"
