import paramiko
import psutil
import gputil as gp
from colorama import Fore, Style

def clear_terminal():
    print("\033[H\033[J")  # ANSI escape codes to clear the terminal screen

def get_gpu_info(ssh):
    stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits")
    gpu_info = stdout.read().decode("utf-8").splitlines()
    return gpu_info

def get_running_python_processes(hostname, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname, username=username, password=password)
        
        gpu_info = get_gpu_info(ssh)
        
        stdin, stdout, stderr = ssh.exec_command("ps aux | grep python")
        processes = stdout.read().decode("utf-8").splitlines()

        running_python_processes = []
        for process in processes:
            if "python" in process:
                running_python_processes.append(process)

        return running_python_processes, gpu_info

    finally:
        ssh.close()

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

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh.connect(hostname, username=username, password=password)
            running_processes, gpu_info = get_running_python_processes(hostname, username, password)

            print(f"Running Python processes on {Fore.BLUE}{hostname}{Style.RESET_ALL}:")
            for process in running_processes:
                print(process)

            print("GPU Information:")
            for info in gpu_info:
                print(info)

        finally:
            ssh.close()

if __name__ == "__main__":
    main()
