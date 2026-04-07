# Running HPEMA on ARC

This guide walks through getting HPEMA running on Virginia Tech's ARC cluster from scratch. Follow each step in order.

---

## 1. Log into ARC

```bash
ssh pid@falcon1.arc.vt.edu
```

Replace `pid` with your VT login ID (the one you use for Hokies, Canvas, etc. — not the 9-digit number).

**Password prompt:** Enter your VT password immediately followed by a comma and the 6-digit OTP from the Duo Mobile app, with no spaces.

```
Password: yourpassword,123456
```

Open Duo Mobile, find the Virginia Tech entry, and type the code shown before it expires. Press Enter.

You are now on a Falcon login node.

---

## 2. Reserve a GPU Node

```bash
salloc --account=muataz --gres=gpu:1 --cpus-per-task=16 --partition=l40s_normal_q
```

Wait for the reservation to be granted. This can take anywhere from a few seconds to a few minutes depending on cluster load. When ready, you will see something like:

```
salloc: Nodes fal038 are ready for job
```

Note the node name (`fal038` in this example — yours may differ).

---

## 3. SSH into the GPU Node

```bash
ssh fal038
```

Use whatever node name was shown in the previous step. Verify you have a GPU:

```bash
nvidia-smi
```

You should see an L40S (or similar) listed. If this command errors, you are not on a GPU node — go back to step 2.

---

## 4. Activate the Project Environment

```bash
cd /projects/meng/Capstone
source .venv/bin/activate
```

---

## 5. Start HPEMA

```bash
bash ml/slurm/start_tmux.sh
```

This script will:

1. Install any missing Python dependencies from `requirements.txt`
2. Install vLLM if not already present **(expect 2–6 minutes on first run)**
3. Launch a tmux session with two panes side by side

| Pane | What it does |
|------|-------------|
| Left | vLLM — loads and serves the language model |
| Right | HPEMA CLI — waits for vLLM before accepting input |

---

## 6. Wait for vLLM

Watch the **left pane**. vLLM prints a lot of startup output while it loads the model weights. It is ready when output stops and you see a line like:

```
INFO:     Application startup complete.
```

> If the right pane says "vLLM not found" — ignore it. The CLI checks early and vLLM sometimes takes longer than the wait window. Once vLLM shows the ready message, the CLI will work.

---

## 7. Use the CLI

Click into the **right pane** and send a message. If HPEMA replies, everything is working.

---

## Tmux Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+b` then arrow key | Move between panes |
| `Ctrl+b` then `z` | Maximize / unmaximize the current pane |
| `Ctrl+b` then `x`, then `y` | Kill the current pane |

---

## 8. Shutting Down Properly

Freeing the GPU node correctly matters — leaving it allocated wastes cluster resources for the team.

**Step 1 — Kill both tmux panes:**

With focus on the right pane: `Ctrl+b` → `x` → `y`

With focus on the left pane: `Ctrl+b` → `x` → `y`

tmux will exit automatically once both panes are closed.

**Step 2 — Release the GPU reservation:**

Type `exit` repeatedly until you see a message like:

```
salloc: Job allocation XXXXXXX has been revoked.
```

This confirms the GPU node has been freed.

**Step 3 — Exit ARC:**

Type `exit` one final time to disconnect from the Falcon login node.
