#!/bin/bash
# two profiling windows on the serving production: C1 and C4 decode, 96 new tokens each, prompt ~1K
H=gx10-1; API=http://127.0.0.1:8210
prompt=$(python3 -c "print('The history of numerical linear algebra begins with ' * 120)")
req() { curl -s -m 300 $API/v1/completions -H 'content-type: application/json' -d "{\"model\":\"glm-5.2-quanttrio\",\"prompt\":\"$1\",\"max_tokens\":96,\"temperature\":0,\"stream\":false}" -o /dev/null -w "%{http_code} %{time_total}s\n"; }
export -f req; export API
# warm (prefix cache + graphs) outside the window
ssh $H "$(declare -f req); API=$API; req '$prompt'"
for win in C1 C4; do
  n=1; [ $win = C4 ] && n=4
  ssh $H "curl -s -X POST $API/start_profile -o /dev/null -w 'start_profile %{http_code}\n'"
  ssh $H "$(declare -f req); API=$API; for i in \$(seq 1 $n); do req '$prompt variant '\$i & done; wait"
  ssh $H "curl -s -X POST $API/stop_profile -o /dev/null -w 'stop_profile %{http_code}\n'"
  sleep 20   # let ranks flush traces
  for ip in gx10-1 192.168.100.12 192.168.100.13 192.168.100.14; do ssh -o BatchMode=yes $ip "cd /var/tmp/glm-legacy/hf/profiles 2>/dev/null && for f in *.json.gz; do [ -f \$f ] && case \$f in ${win}_*) ;; *) mv \$f ${win}_\$f;; esac; done; ls -la /var/tmp/glm-legacy/hf/profiles | tail -n +2 | awk '{print \$5, \$9}'" 2>/dev/null; done
done
