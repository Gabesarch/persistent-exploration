#!/usr/bin/env python3
"""Read active evaluation output identities without executing submitted wrappers."""
from pathlib import Path
import shlex
import subprocess


def identities(snapshot):
    for line in snapshot.splitlines():
        name, working_directory, command = line.split('|', 2)
        yield 'NAME', name
        if working_directory:
            yield 'OUTPUT', str(Path(working_directory).resolve())
        try:
            script = Path(shlex.split(command)[0])
            if not script.is_file() or script.stat().st_size > 1024 * 1024:
                continue
            for source in script.read_text().splitlines():
                if not source.startswith('export OUTPUT_ROOT='):
                    continue
                words = shlex.split(source)
                if len(words) != 2:
                    continue
                output = words[1].split('=', 1)[1]
                if output.startswith('/') and not any(c in output for c in '$`'):
                    yield 'OUTPUT', str(Path(output).resolve())
        except (OSError, ValueError, IndexError):
            continue


if __name__ == '__main__':
    result = subprocess.run(['squeue', '-u', 'gs1693', '-h', '-r', '-t', 'PD,R,CF,CG,S',
                             '-o', '%j|%Z|%o'], check=True, capture_output=True,
                            text=True, timeout=150)
    for kind, value in identities(result.stdout):
        print(kind + '\t' + value)
