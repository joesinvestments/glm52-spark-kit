#!/usr/bin/env bash
# node_protect.sh: make a DGX Spark node survive host-memory exhaustion without a
# power button. Run ON the node as a sudo-capable user. Idempotent. Proves each layer.
#
#  1. earlyoom: kill the largest process at MIN_FREE_PCT free RAM / swap, before
#     the kernel OOM path lets sshd starve. Never targets sshd, systemd, dockerd.
#  2. sshd (and dockerd) protected: OOMScoreAdjust=-1000, MemoryMin so pages stay.
#  3. kernel watchdog daemon on the SBSA Generic Watchdog: pets /dev/watchdog; if
#     userland stops answering (load1 > WD_MAX_LOAD1 or free pages < WD_MIN_MEM_PAGES,
#     or the daemon itself cannot run) the SoC reboots the node in 60 s. Replaces the
#     physical power cycle. Do NOT add a ping test: the daemon's raw-socket ping to
#     127.0.0.1 fails on this kernel and forces a clean reboot after 80 s.
#
# Honest limit: earlyoom did not fire during the 2026-08-17 thrash (page cache of
# mmapped weights counts as available); the watchdog reboot is the layer that works.
set -euo pipefail
MIN_FREE_PCT="${MIN_FREE_PCT:-4}"        # earlyoom -m
MIN_SWAP_PCT="${MIN_SWAP_PCT:-100}"      # earlyoom -s: 100 = trigger on RAM alone; swap fills too late here
WD_MAX_LOAD1="${WD_MAX_LOAD1:-96}"       # watchdog: 1-min load that means "thrashing"
WD_MIN_MEM_PAGES="${WD_MIN_MEM_PAGES:-65536}"  # ~256 MiB of pages free, else reboot
say() { printf '\n== %s\n' "$*"; }

say "1/3 earlyoom"
if ! command -v earlyoom >/dev/null; then sudo -n env DEBIAN_FRONTEND=noninteractive NEEDRESTART_SUSPEND=1 apt-get install -y -qq earlyoom >/dev/null 2>&1; fi
sudo -n tee /etc/default/earlyoom >/dev/null <<EOF
EARLYOOM_ARGS="-m $MIN_FREE_PCT -s $MIN_SWAP_PCT -r 60 --avoid ^(sshd|systemd|dockerd|containerd|init)$ --prefer ^(python3|VLLM|vllm|ray::) -n"
EOF
sudo -n systemctl enable earlyoom >/dev/null 2>&1; sudo -n systemctl restart earlyoom
systemctl is-active earlyoom && sudo -n journalctl -u earlyoom -n 2 --no-pager | tail -1

say "2/3 sshd + dockerd OOM immunity"
for svc in ssh docker; do
  sudo -n mkdir -p /etc/systemd/system/$svc.service.d
  sudo -n tee /etc/systemd/system/$svc.service.d/oom-protect.conf >/dev/null <<EOF
[Service]
OOMScoreAdjust=-1000
OOMPolicy=continue
MemoryMin=256M
EOF
done
sudo -n systemctl daemon-reload
sudo -n systemctl restart ssh   # existing sessions survive a restart
for svc in ssh docker; do echo "$svc: $(systemctl show $svc -p OOMScoreAdjust -p MemoryMin | tr '\n' ' ')"; done
echo "sshd MainPID oom_score_adj: $(cat /proc/$(systemctl show ssh -p MainPID --value)/oom_score_adj)"

say "3/3 kernel watchdog daemon"
if [ ! -c /dev/watchdog ]; then echo "no /dev/watchdog on this SoC; kernel watchdog layer skipped"; else
  if ! command -v watchdog >/dev/null; then sudo -n env DEBIAN_FRONTEND=noninteractive NEEDRESTART_SUSPEND=1 apt-get install -y -qq watchdog >/dev/null 2>&1; fi
  sudo -n tee /etc/watchdog.conf >/dev/null <<EOF
watchdog-device = /dev/watchdog
watchdog-timeout = 60
interval = 10
realtime = yes
priority = 1
max-load-1 = $WD_MAX_LOAD1
min-memory = $WD_MIN_MEM_PAGES
log-dir = /var/log/watchdog
EOF
  sudo -n mkdir -p /var/log/watchdog
  # the package ships wd_keepalive (pets only, no checks); it grabs /dev/watchdog and blocks the real daemon
  sudo -n systemctl stop wd_keepalive 2>/dev/null; sudo -n systemctl disable wd_keepalive >/dev/null 2>&1
  # headless boxes: plymouth-quit-wait stalled a post-power-cycle boot and blocked watchdog.service from starting
  sudo -n systemctl stop plymouth-quit-wait.service 2>/dev/null; sudo -n systemctl mask plymouth-quit-wait.service >/dev/null 2>&1
  sudo -n systemctl enable watchdog >/dev/null 2>&1; sudo -n systemctl reset-failed watchdog 2>/dev/null; sudo -n systemctl restart watchdog; sleep 8
  systemctl is-active watchdog && sudo -n lsof /dev/watchdog 2>/dev/null | tail -1 && echo "watchdog petting /dev/watchdog (timeout 60s, reboot if load1>$WD_MAX_LOAD1 or free<$WD_MIN_MEM_PAGES pages)"
fi

say "PROOF: synthetic memory hog must be killed by earlyoom, sshd must survive"
python3 - <<'PY' &
import time
b=[]
try:
    while True: b.append(bytearray(512*1024*1024)); time.sleep(0.05)
except MemoryError: time.sleep(30)
PY
hog=$!
for i in $(seq 1 60); do sleep 2; kill -0 $hog 2>/dev/null || break; done
if kill -0 $hog 2>/dev/null; then echo "hog still alive after 120s (earlyoom threshold not reached; free RAM large). Killing hog."; kill $hog; else echo "hog killed at $(date -u +%T)"; fi
sudo -n journalctl -u earlyoom --since "-3min" --no-pager | grep -iE "sending|killed" | tail -2 || true
echo "sshd pid still $(pgrep -o -x sshd): OK"
say "done on $(hostname)"
