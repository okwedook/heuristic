FROM ubuntu:latest

RUN apt-get update && apt-get install -y g++ python3 unzip python3-colorama && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ton_crypto_lib-x86_64-linux.zip ./
RUN unzip ton_crypto_lib-x86_64-linux.zip
COPY run_tests.py ./
COPY tests ./tests

COPY solution.cpp .
RUN g++ -O2 -std=c++17 -Iinclude -Iinclude/crypto -Iinclude/tdutils -Iinclude/ton -Iinclude/common solution.cpp -L. -lton_crypto_lib -o solution

ENV LD_LIBRARY_PATH=.
CMD ["python3", "run_tests.py"]