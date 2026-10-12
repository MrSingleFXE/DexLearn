"""CPU sample all 100 candidates, evaluate in Bench, then report probability Top-k."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', required=True, help='Full folder name beneath DexLearn/output')
    p.add_argument('--bench', type=Path, required=True)
    p.add_argument('--bench-python', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--objects', type=Path, help='Optional DGN_2k root, including scenes, split and point clouds')
    p.add_argument('--steps', type=int, nargs='+', default=[10000,20000,30000,40000,50000])
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--workers', type=int, default=40, help='Physics workers per checkpoint')
    p.add_argument('--expected-scenes', type=int, default=10892)
    a = p.parse_args(); a.bench = a.bench.resolve(); a.output = a.output.resolve()
    if Path(a.experiment).name != a.experiment: raise ValueError('Expected an experiment folder name')
    if len(set(a.steps)) != len(a.steps): raise ValueError('Duplicate steps')
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
    def one(step):
        job = a.output / f'step_{step:06d}'; job.mkdir()
        ckpt = ROOT / 'output' / a.experiment / 'ckpts' / f'step_{step:06d}.pth'
        if not ckpt.is_file(): raise FileNotFoundError(ckpt)
        raw = job / a.experiment / 'tests' / f'step_{step:06d}'
        commands = [
            [sys.executable,'-m','dexlearn.sample','-e',a.experiment,'resume=false',f'ckpt={ckpt}',
             f'output_folder={job}','device=cpu','data.split_path=valid_split','data.test_split=test',
             'data.mini_test=false','data.num_workers=0','seed=0','algo.batch_size=32',
             'algo.test_grasp_num=100','algo.test_topk=100'],
            [a.bench_python,str(a.bench/'script/run_sharpa.py'),'--raw',str(raw),'--output',str(job/'bench'),
             '--format','Learning','--expected-scenes',str(a.expected_scenes),'--candidates-per-scene','100','--workers',str(a.workers)],
            [a.bench_python,str(a.bench/'script/analyze_candidate_ranking.py'),'--raw',str(raw),'--bench',str(job/'bench'),
             '--out',str(job/'analysis'),'--expected-scenes',str(a.expected_scenes),'--workers','4']]
        if a.objects is not None:
            commands[0].append(f'data.object_path={a.objects.resolve()}')
        for stage, command in zip(('sample','physics','ranking'),commands):
            with (job/f'{stage}.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return str(step), json.loads((job/'analysis/ranking_summary.json').read_text())
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results = dict(pool.map(one,a.steps))
    (a.output/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__': main()
