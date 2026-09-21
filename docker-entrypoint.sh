#!/bin/sh
# Starts sshd when the container is handed a PUBLIC_KEY (RunPod sets this from
# the key in your account settings), then hands off to CMD. Without that
# variable nothing changes -- `docker run -it ... scale-transformer` still drops
# you straight into bash with no daemon running.
set -eu

if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys

    # Host keys are generated per container, never baked into the image -- a
    # shared host key would make every pod off this image impersonate the others.
    ssh-keygen -A

    # An ssh session does not inherit the ENVs set in this Dockerfile, so PATH
    # would miss /opt/venv/bin and `python` would be the system one. /etc/environment
    # is read by PAM for non-interactive sessions too (what Zed and `ssh host cmd`
    # use); profile.d covers interactive login shells.
    for v in PATH LANG HF_HOME TRITON_CACHE_DIR UV_PROJECT_ENVIRONMENT UV_PYTHON \
             UV_LINK_MODE UV_COMPILE_BYTECODE PYTHONUNBUFFERED PYTHONDONTWRITEBYTECODE \
             CUDA_HOME LD_LIBRARY_PATH NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES; do
        eval "val=\${$v:-}"
        [ -n "$val" ] || continue
        printf '%s=%s\n' "$v" "$val" >> /etc/environment
        printf 'export %s="%s"\n' "$v" "$val" >> /etc/profile.d/10-container-env.sh
    done

    mkdir -p /var/run/sshd
    /usr/sbin/sshd
fi

exec "$@"
