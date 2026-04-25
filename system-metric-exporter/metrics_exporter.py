import os
import subprocess
import logging
import time
import psutil
from prometheus_client import start_http_server, Gauge
from prometheus_client.exposition import basic_auth_handler

# Logging set up
logging.basicConfig(level=logging.INFO)

# Environment variable for /proc path (default to '/proc')
PROC_PATH = os.getenv("PROC_PATH", "/proc")

# Prometheus metrics
# Disk IO metrics
io_read_rate = Gauge(
    "io_read_rate",
    "Rate of reads from the disk (in reads per second)",
    ["device"],
)
io_write_rate = Gauge(
    "io_write_rate",
    "Rate of writes to the disk (in writes per second)",
    ["device"],
)
io_tps = Gauge("io_tps", "Number of transfers per second", ["device"])
io_read_bytes = Gauge(
    "io_read_bytes", "Total bytes read from the disk", ["device"]
)
io_write_bytes = Gauge(
    "io_write_bytes", "Total bytes written to the disk", ["device"]
)

# CPU metric
cpu_avg_percent = Gauge(
    "cpu_avg_percent",
    "CPU usage by mode (user, nice, system, iowait, idle)",
    ["mode"],
)

# Memory metrics
mem_total = Gauge("mem_total", "Total memory")
mem_free = Gauge("mem_free", "Free memory")
mem_available = Gauge("mem_available", "Available memory")
mem_buffers = Gauge("mem_buffers", "Memory used by buffers")
mem_cached = Gauge("mem_cached", "Memory used by cache")

# Swap metrics
swap_total = Gauge("swap_total", "Total swap memory")
swap_free = Gauge("swap_free", "Free swap memory")
swap_cached = Gauge("swap_cached", "Swap memory used as cache")


# Function to collect disk IO statistics using iostat
def collect_disk_io():
    try:
        result = subprocess.check_output(
            ["iostat", "-d", "-x", "1", "2"]
        ).decode("utf-8")
        lines = result.splitlines()
        for line in lines:
            if "Device" in line:
                continue
            columns = line.split()
            if len(columns) < 11:
                continue

            device = columns[0]
            read_rate = float(columns[3])
            write_rate = float(columns[4])
            tps = float(columns[2])
            read_bytes = float(columns[5])
            write_bytes = float(columns[6])

            io_read_rate.labels(device=device).set(read_rate)
            io_write_rate.labels(device=device).set(write_rate)
            io_tps.labels(device=device).set(tps)
            io_read_bytes.labels(device=device).set(read_bytes)
            io_write_bytes.labels(device=device).set(write_bytes)

        logging.info("Successfully collected disk IO stats")
    except Exception as e:
        logging.error(f"Failed to collect disk IO stats: {e}")


# Function to collect CPU usage statistics
def collect_cpu_usage():
    try:
        cpu_times = psutil.cpu_times_percent(interval=1)
        cpu_avg_percent.labels(mode="user").set(cpu_times.user)
        cpu_avg_percent.labels(mode="system").set(cpu_times.system)
        cpu_avg_percent.labels(mode="idle").set(cpu_times.idle)
        cpu_avg_percent.labels(mode="iowait").set(
            getattr(cpu_times, "iowait", 0)
        )

        logging.info(f"CPU Usage Collected")
    except Exception as e:
        logging.error(f"Failed to collect CPU stats: {e}")


# Function to collect memory and swap stats from meminfo
def collect_memory_stats():
    try:
        with open(f"{PROC_PATH}/meminfo", "r") as f:
            mem_info = f.readlines()

        for line in mem_info:
            fields = line.split(":")
            if len(fields) != 2:
                continue
            key = fields[0].strip()
            value = int(fields[1].strip().split()[0])

            if key == "MemTotal":
                mem_total.set(value)
            elif key == "MemFree":
                mem_free.set(value)
            elif key == "MemAvailable":
                mem_available.set(value)
            elif key == "Buffers":
                mem_buffers.set(value)
            elif key == "Cached":
                mem_cached.set(value)
            elif key == "SwapTotal":
                swap_total.set(value)
            elif key == "SwapFree":
                swap_free.set(value)
            elif key == "SwapCached":
                swap_cached.set(value)

        logging.info("Successfully collected memory and swap stats")
    except Exception as e:
        logging.error(f"Failed to collect memory stats: {e}")


# Main function to start the server and collect metrics
def main():
    start_http_server(18000)
    logging.info(
        "Prometheus metrics server started on http://localhost:18000/metrics"
    )

    while True:
        collect_disk_io()
        collect_cpu_usage()
        collect_memory_stats()
        time.sleep(1)


if __name__ == "__main__":
    main()
