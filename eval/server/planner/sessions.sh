#!/usr/bin/env bash
# manage_sessions.sh -- helper script to manage Natural-Plan evaluation sessions
# Usage:
#   ./manage_sessions.sh list          # show all np_* sessions with GPU info
#   ./manage_sessions.sh kill all      # kill all np_* sessions  
#   ./manage_sessions.sh kill 14b      # kill all 14b model sessions
#   ./manage_sessions.sh kill meeting  # kill all meeting sessions
#   ./manage_sessions.sh status        # show GPU usage and running sessions

set -euo pipefail

ACTION=${1:-list}
TARGET=${2:-}

list_sessions() {
    echo "=== Active Natural-Plan Sessions ==="
    screen -ls | grep "np_" | while read line; do
        session_full=$(echo "$line" | awk '{print $1}')
        screen_pid=${session_full%%.*}
        session=${session_full#*.}
        status=$(echo "$line" | grep -o "(Attached\|Detached)")
        
        # Handle optional prefixes like np_budget_ or np_scale_
        stripped=${session#np_}
        stripped=${stripped#budget_}
        stripped=${stripped#scale_}
        if [[ $stripped =~ ^([^_]+)_(.+)$ ]]; then
            task="${BASH_REMATCH[1]}"
            model="${BASH_REMATCH[2]}"
        else
            task="$stripped"
            model="14b"  
        fi
        
        # Static GPU mapping (hardcoded from bypass.sh and np_budget.sh) ----------------
        case "${model},${task}" in
            # 14B models: GPUs 0,1,2
            "14b,trip") gpu=0;;
            "14b,meeting") gpu=1;;
            "14b,calendar") gpu=2;;
            # 8B models: GPUs 3,4,5
            "8b,trip") gpu=3;;
            "8b,meeting") gpu=4;;
            "8b,calendar") gpu=5;;
            # 1.5B models: GPUs 6,7,7
            "1.5b,trip") gpu=6;;
            "1.5b,meeting") gpu=7;;
            "1.5b,calendar") gpu=7;;
            *) gpu="?";;
        esac

        
        printf "  %-20s | %-8s | %-7s | GPU %-2s | %s\n" "$session" "$task" "$model" "$gpu" "$status"
    done
    
    if ! screen -ls | grep -q "np_"; then
        echo "  No Natural-Plan sessions running."
    fi
}

kill_sessions() {
    local target="$1"
    local killed=0
    
    echo "=== Killing sessions matching: $target ==="
    
    screen -ls | grep "np_" | while read line; do
        session_full=$(echo "$line" | awk '{print $1}')
        screen_pid=${session_full%%.*}
        session=${session_full#*.}
        
        should_kill=false
        
        # Strip prefixes
        stripped=${session#np_}
        stripped=${stripped#budget_}
        stripped=${stripped#scale_}
        if [[ $stripped =~ ^([^_]+)_(.+)$ ]]; then
            session_task="${BASH_REMATCH[1]}"
            session_model="${BASH_REMATCH[2]}"
        else
            session_task="$stripped"
            session_model="14b"
        fi
        
        case "$target" in
            "all")
                should_kill=true;;
            "14b"|"8b"|"1.5b")
                if [[ $session_model == "$target" ]]; then
                    should_kill=true
                fi;;
            "meeting"|"calendar"|"trip")
                if [[ $session_task == "$target" ]]; then
                    should_kill=true
                fi;;
            *)
                if [[ $session == *"$target"* ]]; then
                    should_kill=true
                fi;;
        esac
        
        if $should_kill; then
            echo "  Killing session: $session"
            screen -S "$session" -X quit || echo "    Failed to kill $session"
            ((killed++))
        fi
    done
    
    echo "Killed $killed sessions."
}

show_gpu_status() {
    echo "=== GPU Usage ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        while IFS=, read -r gpu name mem_used mem_total util; do
            printf "GPU %s: %-20s | %4s/%4s MB | %2s%% util\n" "$gpu" "$name" "$mem_used" "$mem_total" "$util"
        done
    else
        echo "nvidia-smi not available"
    fi
    echo ""
}

case "$ACTION" in
    list|ls)
        list_sessions;;
    kill)
        if [[ -z "$TARGET" ]]; then
            echo "Error: kill requires a target (all, 14b, 8b, 1.5b, meeting, calendar, trip)"
            exit 1
        fi
        kill_sessions "$TARGET";;
    status)
        show_gpu_status
        list_sessions;;
    *)
        echo "Usage: $0 {list|kill <target>|status}"
        echo ""
        echo "Examples:"
        echo "  $0 list              # show all sessions"  
        echo "  $0 kill all          # kill all np_* sessions"
        echo "  $0 kill 14b          # kill all 14b sessions"
        echo "  $0 kill meeting      # kill all meeting sessions"
        echo "  $0 status            # show GPU usage and sessions"
        exit 1;;
esac