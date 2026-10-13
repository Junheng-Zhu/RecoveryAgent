"""Offline v0.3-C diagnostic entry point; this CLI has no online/API option."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recovery_agent.hardening.termination import diagnose_directory,compare_stop_protocols
from recovery_agent.hardening.mutation_diagnostics import diagnose_pair

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="outputs/v0.3b_teacher10")
    parser.add_argument("--output",default="outputs/v0.3c_teacher10_diagnostic")
    parser.add_argument("--paired-second",help="Second collection for matched offline mutation attribution")
    parser.add_argument("--compare-third")
    parser.add_argument("--compare-first")
    parser.add_argument("--compare-second")
    args=parser.parse_args()
    if args.compare_first or args.compare_second or args.compare_third:
        if not (args.compare_first and args.compare_second):
            parser.error("Both comparison directories are required")
        result=compare_stop_protocols(args.compare_first,args.compare_second,*([args.compare_third] if args.compare_third else []))
    elif args.paired_second:
        result=diagnose_pair(args.input,args.paired_second,args.output)
    else:
        result=diagnose_directory(args.input,args.output)
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
