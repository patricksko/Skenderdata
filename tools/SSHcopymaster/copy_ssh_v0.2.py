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

        self.clear_button = tk.Button(root, text="Clear All", command=self.clear_all)
        self.clear_button.pack()

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

        self.folder_name_label = tk.Label(root, text="Enter Folder Name:")
        self.folder_name_label.pack()

        self.folder_name_entry = tk.Entry(root)
        self.folder_name_entry.pack()

    def clear_all(self):
        self.file_listbox.delete(0, tk.END)

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
        selected_host = self.host_combobox.get()

        if selected_host not in self.ssh_hosts:
            messagebox.showerror("Invalid Host", "Please select a valid SSH host from the dropdown menu.")
            return

        folder_name = self.folder_name_entry.get()  # Get the specified folder name
        if not folder_name:
            messagebox.showwarning("Missing Folder Name", "Please enter a folder name.")
            return

        host_details = self.ssh_hosts[selected_host]

        with paramiko.Transport((host_details['host'], host_details['port'])) as transport:
            transport.connect(username=host_details['username'], password=host_details['password'])
            sftp = paramiko.SFTPClient.from_transport(transport)

            # Create the folder on the remote host
            try:
                sftp.mkdir(host_details['remote_path'] + '/' + folder_name)
            except:
                pass  # Folder already exists

            # Get all items from the listbox
            all_items = self.file_listbox.get(0, tk.END)

            for item in all_items:
                if os.path.exists(item):  # Check if the path exists (to handle folders)
                    remote_path = os.path.join(host_details['remote_path'], folder_name, os.path.basename(item))
                    total_size = os.path.getsize(item)

                    with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                        def update_progress(transferred, total):
                            pbar.update(transferred - pbar.n)

                        sftp.put(item, remote_path, callback=update_progress)
                        pbar.update(total_size - pbar.n)

                    pbar.close()

        messagebox.showinfo("Transfer Complete", "Files have been transferred successfully.")



    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = SSHFileTransferApp(root)
    app.root.mainloop()