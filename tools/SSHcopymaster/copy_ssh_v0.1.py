import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Progressbar, Combobox
from tqdm import tqdm
import paramiko
import threading
import json
from tkinterdnd2 import DND_FILES, TkinterDnD

class SSHFileTransferApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SSH File Transfer")

        # Create UI elements
        self.file_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)

        self.add_button = tk.Button(root, text="Add Files/Folders", command=self.add_files)
        self.add_button.pack()

        self.transfer_button = tk.Button(root, text="Transfer Files", command=self.transfer_files)
        self.transfer_button.pack()

        self.progress_bar = Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress_bar.pack()

        # Load SSH hosts from config file
        self.ssh_hosts = self.load_ssh_hosts()

        # Create a dropdown menu for SSH hosts
        self.host_label = tk.Label(root, text="Select SSH Host:")
        self.host_label.pack()

        self.host_combobox = Combobox(root, values=list(self.ssh_hosts.keys()))
        self.host_combobox.pack()

        # Bind drag-and-drop handlers using tkinterdnd2
        self.file_listbox.drop_target_register(DND_FILES)
        self.file_listbox.dnd_bind('<<Drop>>', self.handle_drop)

    def load_ssh_hosts(self):
        try:
            with open('config.json', 'r') as config_file:
                return json.load(config_file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def add_files(self):
        files = filedialog.askopenfilenames()
        for file in files:
            self.file_listbox.insert(tk.END, file)

    def drag_enter(self, event):
        event.widget.focus_force()

    def drag_leave(self, event):
        pass

    def handle_drop(self, event):
        if event.data:
            file_paths = event.data.strip().split()  # Split the string into individual paths
            for file in file_paths:
                if os.path.exists(file):  # Check if the path exists (to handle folders)
                    self.file_listbox.insert(tk.END, file)

    def transfer_files(self):
        selected_files = self.file_listbox.curselection()

        if not selected_files:
            messagebox.showwarning("No Files Selected", "Please select files to transfer.")
            return

        selected_files = [self.file_listbox.get(idx) for idx in selected_files]
        selected_host = self.host_combobox.get()

        if selected_host not in self.ssh_hosts:
            messagebox.showerror("Invalid Host", "Please select a valid SSH host from the dropdown menu.")
            return

        host_details = self.ssh_hosts[selected_host]

        with paramiko.Transport((host_details['host'], host_details['port'])) as transport:
            transport.connect(username=host_details['username'], password=host_details['password'])
            sftp = paramiko.SFTPClient.from_transport(transport)

            for file in selected_files:
                remote_path = os.path.join(host_details['remote_path'], os.path.basename(file))
                total_size = os.path.getsize(file)

                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                    def update_progress(transferred, total):  # This is the correct definition
                        pbar.update(transferred - pbar.n)

                    sftp.put(file, remote_path, callback=update_progress)
                    pbar.update(total_size - pbar.n)

                pbar.close()

        messagebox.showinfo("Transfer Complete", "Files have been transferred successfully.")


    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = SSHFileTransferApp(root)
    app.run()