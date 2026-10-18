#!/bin/bash
# Mac resource watchdog for yt-distill jobs (16 GB M4). Prints one line per state change.
# At "critical" it pauses (SIGSTOP) the offending processes and resumes them (SIGCONT) on recovery:
#   memory/swap critical → pause GPU/model jobs (whisper-cli, mlx_worker, model-loading python)
#   disk critical        → pause downloads (hf, curl, yt-dlp)
# Usage: scripts/resource_watch.sh [poll_seconds]
POLL=${1:-10}
MEM_WARN=25; MEM_CRIT=12     # % memory free (memory_pressure)
SWAP_WARN=2048; SWAP_CRIT=6144   # MB swap used
DISK_WARN=20; DISK_CRIT=10   # GB free on the data volume
GPU_PAT='whisper-cli|mlx_worker|ic_test.py|indic_transcribe|onnxruntime'
DL_PAT='hf download|huggingface_hub hf|curl -fL|yt-dlp'

level() { # value warn crit lower_is_worse
  if [ "$4" = 1 ]; then
    [ "$1" -le "$3" ] && echo crit && return; [ "$1" -le "$2" ] && echo warn && return
  else
    [ "$1" -ge "$3" ] && echo crit && return; [ "$1" -ge "$2" ] && echo warn && return
  fi
  echo ok
}
signal() { pgrep -f "$1" | while read -r p; do kill "-$2" "$p" 2>/dev/null; done; }

prev=""; gpu_paused=0; dl_paused=0
while true; do
  mem=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub("%","",$2); print int($2)}')
  swap=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
  disk=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  therm=$(pmset -g therm 2>/dev/null | awk -F'= ' '/CPU_Speed_Limit/{print $2}'); therm=${therm:-100}
  lm=$(level "$mem" $MEM_WARN $MEM_CRIT 1); ls=$(level "$swap" $SWAP_WARN $SWAP_CRIT 0)
  ld=$(level "$disk" $DISK_WARN $DISK_CRIT 1); lt=ok; [ "$therm" -lt 100 ] && lt=warn
  state="mem=$lm swap=$ls disk=$ld thermal=$lt"

  if [ "$lm" = crit ] || [ "$ls" = crit ]; then
    [ $gpu_paused = 0 ] && signal "$GPU_PAT" STOP && gpu_paused=1 && echo "$(date +%T) ACTION paused GPU/model jobs (mem ${mem}% free, swap ${swap}MB)"
  elif [ $gpu_paused = 1 ] && [ "$lm" = ok ] && [ "$ls" != crit ]; then
    signal "$GPU_PAT" CONT; gpu_paused=0; echo "$(date +%T) ACTION resumed GPU/model jobs (mem ${mem}% free)"
  fi
  if [ "$ld" = crit ]; then
    [ $dl_paused = 0 ] && signal "$DL_PAT" STOP && dl_paused=1 && echo "$(date +%T) ACTION paused downloads (disk ${disk}GB free)"
  elif [ $dl_paused = 1 ] && [ "$ld" = ok ]; then
    signal "$DL_PAT" CONT; dl_paused=0; echo "$(date +%T) ACTION resumed downloads (disk ${disk}GB free)"
  fi

  if [ "$state" != "$prev" ]; then
    echo "$(date +%T) $state | mem ${mem}% free, swap ${swap}MB, disk ${disk}GB free, cpu limit ${therm}%, whisper procs $(pgrep whisper-cli | wc -l | tr -d ' ')"
    prev=$state
  fi
  sleep "$POLL"
done
