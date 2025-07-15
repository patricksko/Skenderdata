import paramiko
import psutil
from colorama import Fore, Style

def clear_terminal():
    print("\033[H\033[J")  # ANSI escape codes to clear the terminal screen


def get_running_python_processes(hostname, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname, username=username, password=password)
        stdin, stdout, stderr = ssh.exec_command("ps aux | grep python")
        processes = stdout.read().decode("utf-8").splitlines()

        running_python_processes = []
        for process in processes:
            if "python" in process:
                running_python_processes.append(process)

        return running_python_processes

    finally:
        ssh.close()

def get_local_running_python_processes():
    processes = psutil.process_iter(attrs=["pid", "name", "cmdline"])

    running_python_processes = []
    for process in processes:
        try:
            if "python" in process.info["name"].lower() and process.info["cmdline"]:
                cmdline = " ".join(process.info["cmdline"])
                running_python_processes.append(f"{process.info['pid']} - {cmdline}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return running_python_processes


def main():

    remote_pcs = [
        {"hostname": "rebecca.acin.tuwien.ac.at", "username": "peter", "password": "hoenig1996"},
        {"hostname": "roxanne.acin.tuwien.ac.at", "username": "peter", "password": "hoenig1996"},
        {"hostname": "128.131.86.156", "username": "peter", "password": "hoenig1996"},
        # Add more remote PCs here
    ]

    clear_terminal()

    for pc in remote_pcs:
        hostname = pc["hostname"]
        username = pc["username"]
        password = pc["password"]

        running_processes = get_running_python_processes(hostname, username, password)

        print(f"Running Python processes on {Fore.BLUE}{hostname}{Style.RESET_ALL}:")
        for process in running_processes:
            print(process)

    local_running_processes = get_local_running_python_processes()

    print(f"Running Python processes on {Fore.GREEN}local host{Style.RESET_ALL}:")
    for process in local_running_processes:
        print(process)
        


if __name__ == "__main__":
    main()