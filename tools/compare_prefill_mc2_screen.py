#!/usr/bin/env python3
"""Read and compare the frozen MC2 on/off screen; no NPU work or file writes."""
import hashlib,json
from pathlib import Path
root=Path('/cache/cch/state-reduction-qwen38-drrqr-aligned-20260908/dk64-energy/preflight/prefill-mc2-screen-v1')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
loaded={}
artifact_hashes={}
for side in ('dk64-on','dk64-off'):
    d=root/side
    loaded[side]={f:json.loads((d/f).read_text()) for f in ('service-manifest.json','activation-smoke.json','performance-screen.json','performance.run1.json')}
    artifact_hashes[side]={f:digest(d/f) for f in ('service-manifest.json','activation-smoke.json','evidence.jsonl','performance-screen.json','performance.run1.json')}
    x=loaded[side]
    m,s,p,r=(x[f] for f in ('service-manifest.json','activation-smoke.json','performance-screen.json','performance.run1.json'))
    assert p['status']=='screen_complete' and s['status']=='activation_smoke_pass'
    assert p['service_manifest_sha256']==s['service_manifest_sha256']==digest(d/'service-manifest.json')
    assert p['activation_smoke_sha256']==digest(d/'activation-smoke.json')
    assert p['report_sha256']==digest(d/'performance.run1.json') and p['summary']==r['summary']
    assert m['prefill_mc2']==s['prefill_mc2']==p['prefill_mc2']==(side=='dk64-on')
    assert len(r['results'])==40 and sorted(v['index'] for v in r['results'])==list(range(40))
    assert all(v['ok'] and v['finish_reason']=='length' and v['output_tokens']==v['requested_output_tokens']==1024 for v in r['results'])
    assert len(r['warmup_results'])==2 and all(v['ok'] and v['output_tokens']==1 for v in r['warmup_results'])
a,b=loaded['dk64-on'],loaded['dk64-off']
manifest_keys=('variant','layout','plugin_version','command','profiler_config','installed_plugin','module_sha256','build_manifest_sha256','wheel_sha256','source_sha256','reference_manifest_sha256','plan_sha256','wrapper_sha256','physical_npu_contract','container_npu_namespace','diagnostic_determinism','inherited_runtime_environment')
assert all(a['service-manifest.json'][k]==b['service-manifest.json'][k] for k in manifest_keys)
flags=[{k:v for k,v in x['service-manifest.json']['plugin_flags'].items() if k not in ('VLLM_ASCEND_DRRQR_PREFILL_MC2','VLLM_ASCEND_DRRQR_EVIDENCE_FILE')} for x in (a,b)]
assert flags[0]==flags[1]
assert a['performance.run1.json']['config']==b['performance.run1.json']['config']
assert a['performance-screen.json']['driver_sha256']==b['performance-screen.json']['driver_sha256']
assert a['performance-screen.json']['client_sha256']==b['performance-screen.json']['client_sha256']
sa,sb=a['activation-smoke.json'],b['activation-smoke.json']
assert sa['script_sha256']==sb['script_sha256'] and sa['driver_sha256']==sb['driver_sha256']
assert len(sa['requests'])==len(sb['requests'])==3
smoke_match=all(all(x[k]==y[k] for k in ('input_sha256','output_sha256','input_tokens','output_tokens','finish_reason')) for x,y in zip(sa['requests'],sb['requests']))
metrics=('output_tokens_per_second','ttft_mean_ms','tpot_mean_ms','benchmark_duration_s')
comp={k:{'off':b['performance.run1.json']['summary'][k],'on':a['performance.run1.json']['summary'][k],'change_percent':100*(a['performance.run1.json']['summary'][k]/b['performance.run1.json']['summary'][k]-1)} for k in metrics}
print(json.dumps({'schema':'drrqr-prefill-mc2-pair-analysis/v1','status':'single_pair_complete','artifact_sha256':artifact_hashes,'matched_manifest_fields':manifest_keys,'matching_flags_except_treatment_and_evidence_path':True,'matching_benchmark_config':a['performance.run1.json']['config'],'smoke_three_output_hashes_match':smoke_match,'summary':comp,'exact_output_tokens_per_side':40960,'requests_per_side':40,'activation':{'on':sa['activation'],'off':sb['activation']},'claim':'paired screening only; no repeatability, quality parity, or unpruned-baseline gain demonstrated'},indent=2))

