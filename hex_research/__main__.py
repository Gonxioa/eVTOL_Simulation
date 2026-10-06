import argparse
from pathlib import Path
from .configuration import load,ROOT

def main():
    p=argparse.ArgumentParser(description='Research-only audited six-rotor workflow')
    p.add_argument('command',choices=['audit','validate','faults','generate','report'])
    p.add_argument('--out',type=Path,default=ROOT/'output_hex_audit')
    p.add_argument('--groups',type=int,default=2); p.add_argument('--duration',type=float,default=6)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True); cfg=load()
    if a.command=='validate':
        from .validation import normal
        normal(cfg,a.out)
    elif a.command=='faults':
        from .validation import faults
        faults(cfg,a.out)
    elif a.command=='audit':
        from hex_audit_reports import run
        run(cfg,a.out)
    elif a.command=='generate':
        from .data import generate
        generate(cfg,a.out,a.groups,a.duration)
    elif a.command=='report':
        from hex_audit_reports import report
        report(cfg,a.out)

if __name__=='__main__': main()
