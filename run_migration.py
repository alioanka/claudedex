import os
import time
import subprocess

retries = 5
delay = 5

for i in range(retries):
    result = subprocess.run(
        "alembic revision --autogenerate -m 'Initial baseline migration'",
        shell=True,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("Alembic migration generated successfully.")
        print(result.stdout)
        exit(0)
    else:
        print(f"Attempt {i+1}/{retries} failed. Retrying in {delay} seconds...")
        print(result.stderr)
        time.sleep(delay)

print("Failed to generate Alembic migration after multiple retries.")
exit(1)
