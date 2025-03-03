import base64
import binascii
import os
import subprocess
from typing import Optional
from colorama import Fore, Style, init


def get_base64_size(s: str) -> Optional[int]:
    try:
        decoded = base64.b64decode(s, validate=True)
    except binascii.Error:
        return None
    return len(decoded)


init(autoreset=True)

test_files = os.listdir("tests/")
test_files.sort()
n_tests = len(test_files)
if n_tests == 0:
    print("Error: no tests")
    exit(2)

n_ok = 0
total_points = 0

name_width = max(max(len(s) for s in test_files), 4)

f = open("res/raw_huffman_data.txt", "w")
f.close()

print(f"{Fore.YELLOW}====================== Running {n_tests} tests ======================{Style.RESET_ALL}")
print(f"{Fore.YELLOW}Idx  {"Name":<{name_width}} {"Size":>7}   Compression   Decompression {Style.RESET_ALL}")

for i, test_file in enumerate(test_files):
    i += 1
    with open("tests/" + test_file, "r") as f:
        original_block = f.read().split()[1]
    original_size = get_base64_size(original_block)
    assert original_size is not None
    assert original_size <= 2 ** 21
    print(f"{Fore.YELLOW}{i:>4}{Style.RESET_ALL}",
          f"{test_file:<{name_width}}",
          f"{Fore.CYAN}{original_size:>7}{Style.RESET_ALL}",
          end="   ")

    try:
        result = subprocess.run("./solution", input="compress\n" + original_block, text=True,
                                timeout=2.0, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        encode_stderr = result.stderr
    except subprocess.TimeoutExpired as e:
        print(f"{Fore.RED}TL timeout expired{Style.RESET_ALL}")
        print(e.stderr.decode())
        continue
    if result.returncode != 0:
        print(f"{Fore.RED}RE exitcode={result.returncode}{Style.RESET_ALL}")
        print(result.stderr)
        continue
    compressed_block = result.stdout.strip()
    compressed_size = get_base64_size(compressed_block)
    if compressed_size is None:
        print(f"{Fore.RED}PE invalid base64{Style.RESET_ALL}")
        continue
    if compressed_size > 2 ** 21:
        print(f"{Fore.RED}PE too big output ({compressed_size}){Style.RESET_ALL}")
        continue
    print(f"{Fore.GREEN}OK {Fore.CYAN}{compressed_size:>7}{Style.RESET_ALL}", end="    ")

    try:
        result = subprocess.run("./solution", input="decompress\n" + compressed_block, text=True,
                                timeout=2.0, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        decode_stderr = result.stderr
    except subprocess.TimeoutExpired:
        print(f"{Fore.RED}TL timeout expired{Style.RESET_ALL}")
        continue
    if result.returncode != 0:
        print(f"{Fore.RED}RE exitcode={result.returncode}{Style.RESET_ALL}")
        print("Encode stderr:", encode_stderr)
        print("Decode stderr:", result.stderr)
        continue
    decompressed_block = result.stdout.strip()
    if decompressed_block != original_block:
        print(f"{Fore.RED}WA wrong decompressed block{Style.RESET_ALL}")
        print("Encode stderr:", encode_stderr)
        print("Decode stderr:", result.stderr)
        continue

    points = 1000 * (2 * original_size) / (original_size + compressed_size)
    print(f"{Fore.GREEN}OK{Style.RESET_ALL} {Fore.CYAN}{points:9.3f}{Fore.CYAN}")
    n_ok += 1
    total_points += points

    print("Encode stderr:", encode_stderr)
    print("Decode stderr:", decode_stderr)

if n_ok == n_tests:
    passed_color = Fore.GREEN
else:
    passed_color = Fore.RED
print(f"{Fore.YELLOW}Passed tests:   {passed_color}{n_ok}/{n_tests}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}Average points: {Fore.CYAN}{total_points / n_tests:.3f}{Style.RESET_ALL}")
print(f"{Fore.YELLOW}Total points:   {Fore.CYAN}{total_points:.3f}{Style.RESET_ALL}")
